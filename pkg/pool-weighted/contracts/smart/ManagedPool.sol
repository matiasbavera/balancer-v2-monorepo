// SPDX-License-Identifier: GPL-3.0-or-later
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

pragma solidity ^0.7.0;
pragma experimental ABIEncoderV2;

import "@balancer-labs/v2-solidity-utils/contracts/openzeppelin/ReentrancyGuard.sol";
import "@balancer-labs/v2-solidity-utils/contracts/helpers/ERC20Helpers.sol";
import "@balancer-labs/v2-solidity-utils/contracts/helpers/WordCodec.sol";
import "@balancer-labs/v2-solidity-utils/contracts/helpers/ArrayHelpers.sol";

import "@balancer-labs/v2-pool-utils/contracts/ProtocolFeeCache.sol";

import "../BaseWeightedPool.sol";
import "../WeightedPoolUserData.sol";
import "./WeightCompression.sol";

/**
 * @dev Weighted Pool with mutable tokens and weights, designed to be used in conjunction with a pool controller
 * contract (as the owner, containing any specific business logic). Since the pool itself permits "dangerous"
 * operations, it should never be deployed with an EOA as the owner.
 *
 * Pool controllers can add functionality: for example, allow the effective "owner" to be transferred to another
 * address. (The actual pool owner is still immutable, set to the pool controller contract.) Another pool owner
 * might allow fine-grained permissioning of protected operations: perhaps a multisig can add/remove tokens, but
 * a third-party EOA is allowed to set the swap fees.
 *
 * Pool controllers might also impose limits on functionality so that operations that might endanger LPs can be
 * performed more safely. For instance, the pool by itself places no restrictions on the duration of a gradual
 * weight change, but a pool controller might restrict this in various ways, from a simple minimum duration,
 * to a more complex rate limit.
 *
 * Pool controllers can also serve as intermediate contracts to hold tokens, deploy timelocks, consult with other
 * protocols or on-chain oracles, or bundle several operations into one transaction that re-entrancy protection
 * would prevent initiating from the pool contract.
 *
 * Managed Pools and their controllers are designed to support many asset management use cases, including: large
 * token counts, rebalancing through token changes, gradual weight or fee updates, circuit breakers for
 * IL-protection, and more.
 */
contract ManagedPool is BaseWeightedPool, ProtocolFeeCache, ReentrancyGuard {
    // ManagedPool weights can change over time: these periods are expected to be long enough (e.g. days)
    // that any timestamp manipulation would achieve very little.
    // solhint-disable not-rely-on-time

    using FixedPoint for uint256;
    using WordCodec for bytes32;
    using WeightCompression for uint256;
    using WeightedPoolUserData for bytes;

    // State variables

    // The upper bound is WeightedMath.MAX_WEIGHTED_TOKENS, but this is constrained by other factors, such as Pool
    // creation gas consumption.
    uint256 private constant _MAX_MANAGED_TOKENS = 38;

    uint256 private constant _MAX_MANAGEMENT_SWAP_FEE_PERCENTAGE = 1e18; // 100%

    // Use the _miscData slot in BasePool
    // First 64 bits are reserved for the swap fee
    //
    // Store non-token-based values:
    // Start/end timestamps for gradual weight update
    // Cache total tokens
    // [ 64 bits  | 119 bits |    1 bit    |  32 bits  |   32 bits  |    7 bits    |   1 bit   ]
    // [ reserved |  unused  | restrict LP | end time  | start time | total tokens | swap flag ]
    // |MSB                                                                                 LSB|
    uint256 private constant _SWAP_ENABLED_OFFSET = 0;
    uint256 private constant _TOTAL_TOKENS_OFFSET = 1;
    uint256 private constant _START_TIME_OFFSET = 8;
    uint256 private constant _END_TIME_OFFSET = 40;
    uint256 private constant _MUST_ALLOWLIST_LPS_OFFSET = 72;

    // 7 bits is enough for the token count, since _MAX_MANAGED_TOKENS is 50

    // Store scaling factor and start/end denormalized weights for each token
    // Mapping should be more efficient than trying to compress it further
    // [ 123 bits |  5 bits  |  64 bits   |   64 bits    |
    // [ unused   | decimals | end denorm | start denorm |
    // |MSB                                           LSB|
    mapping(IERC20 => bytes32) private _tokenState;

    // Denormalized weights are stored using the WeightCompression library as a percentage of the maximum absolute
    // denormalized weight: independent of the current _denormWeightSum, which avoids having to recompute the denorm
    // weights as the sum changes.
    uint256 private constant _MAX_DENORM_WEIGHT = 1e22; // FP 10,000

    uint256 private constant _START_DENORM_WEIGHT_OFFSET = 0;
    uint256 private constant _END_DENORM_WEIGHT_OFFSET = 64;
    uint256 private constant _DECIMAL_DIFF_OFFSET = 128;

    // If mustAllowlistLPs is enabled, this is the list of addresses allowed to join the pool
    mapping(address => bool) private _allowedAddresses;

    // We need to work with normalized weights (i.e. they should add up to 100%), but storing normalized weights
    // would require updating all weights whenever one of them changes, for example in an add or remove token
    // operation. Instead, we keep track of the sum of all denormalized weights, and dynamically normalize them
    // for I/O by multiplying or dividing by the `_denormWeightSum`.
    //
    // In this contract, "weights" mean normalized weights, and "denormWeights" refer to how they are stored internally.
    uint256 private _denormWeightSum;

    // Percentage of swap fees that are allocated to the Pool owner, after protocol fees
    uint256 private _managementSwapFeePercentage;

    // Event declarations

    event GradualWeightUpdateScheduled(
        uint256 startTime,
        uint256 endTime,
        uint256[] startWeights,
        uint256[] endWeights
    );
    event SwapEnabledSet(bool swapEnabled);
    event MustAllowlistLPsSet(bool mustAllowlistLPs);
    event ManagementFeePercentageChanged(uint256 managementFeePercentage);
    event AllowlistAddressAdded(address indexed member);
    event AllowlistAddressRemoved(address indexed member);
    event TokenAdded(IERC20 indexed token, uint256 weight, uint256 initialBalance);

    struct NewPoolParams {
        string name;
        string symbol;
        IERC20[] tokens;
        uint256[] normalizedWeights;
        address[] assetManagers;
        uint256 swapFeePercentage;
        bool swapEnabledOnStart;
        bool mustAllowlistLPs;
        uint256 protocolSwapFeePercentage;
        uint256 managementSwapFeePercentage;
    }

    constructor(
        NewPoolParams memory params,
        IVault vault,
        address owner,
        uint256 pauseWindowDuration,
        uint256 bufferPeriodDuration
    )
        BaseWeightedPool(
            vault,
            params.name,
            params.symbol,
            params.tokens,
            params.assetManagers,
            params.swapFeePercentage,
            pauseWindowDuration,
            bufferPeriodDuration,
            owner
        )
        ProtocolFeeCache(vault, params.protocolSwapFeePercentage)
    {
        uint256 totalTokens = params.tokens.length;
        InputHelpers.ensureInputLengthMatch(totalTokens, params.normalizedWeights.length, params.assetManagers.length);

        _setMiscData(_getMiscData().insertUint7(totalTokens, _TOTAL_TOKENS_OFFSET));

        // Double check it fits in 7 bits
        _require(_getTotalTokens() == totalTokens, Errors.MAX_TOKENS);

        // Validate and set initial fee
        _setManagementSwapFeePercentage(params.managementSwapFeePercentage);

        // Initialize the denorm weight sum to the initial normalized weight sum of ONE
        _denormWeightSum = FixedPoint.ONE;

        uint256 currentTime = block.timestamp;
        _startGradualWeightChange(
            currentTime,
            currentTime,
            params.normalizedWeights,
            params.normalizedWeights,
            params.tokens
        );

        // If false, the pool will start in the disabled state (prevents front-running the enable swaps transaction).
        _setSwapEnabled(params.swapEnabledOnStart);

        // If true, only addresses on the manager-controlled allowlist may join the pool.
        _setMustAllowlistLPs(params.mustAllowlistLPs);
    }

    /**
     * @dev Returns true if swaps are enabled.
     */
    function getSwapEnabled() public view returns (bool) {
        return _getMiscData().decodeBool(_SWAP_ENABLED_OFFSET);
    }

    /**
     * @dev Returns true if the allowlist for LPs is enabled.
     */
    function getMustAllowlistLPs() public view returns (bool) {
        return _getMiscData().decodeBool(_MUST_ALLOWLIST_LPS_OFFSET);
    }

    /**
     * @dev Verifies that a given address is allowed to hold tokens.
     */
    function isAllowedAddress(address member) public view returns (bool) {
        return !getMustAllowlistLPs() || _allowedAddresses[member];
    }

    /**
     * @dev Returns the management swap fee percentage as a 18-decimals fixed point number.
     */
    function getManagementSwapFeePercentage() public view returns (uint256) {
        return _managementSwapFeePercentage;
    }

    /**
     * @dev Return start time, end time, and endWeights as an array.
     * Current weights should be retrieved via `getNormalizedWeights()`.
     */
    function getGradualWeightUpdateParams()
        external
        view
        returns (
            uint256 startTime,
            uint256 endTime,
            uint256[] memory endWeights
        )
    {
        // Load current pool state from storage
        bytes32 poolState = _getMiscData();

        startTime = poolState.decodeUint32(_START_TIME_OFFSET);
        endTime = poolState.decodeUint32(_END_TIME_OFFSET);

        (IERC20[] memory tokens, , ) = getVault().getPoolTokens(getPoolId());
        uint256 totalTokens = tokens.length;

        endWeights = new uint256[](totalTokens);

        for (uint256 i = 0; i < totalTokens; i++) {
            endWeights[i] = _normalizeWeight(
                _tokenState[tokens[i]].decodeUint64(_END_DENORM_WEIGHT_OFFSET).uncompress64(_MAX_DENORM_WEIGHT)
            );
        }
    }

    function _getMaxTokens() internal pure virtual override returns (uint256) {
        return _MAX_MANAGED_TOKENS;
    }

    function _getTotalTokens() internal view virtual override returns (uint256) {
        return _getMiscData().decodeUint7(_TOTAL_TOKENS_OFFSET);
    }

    /**
     * @dev Schedule a gradual weight change, from the current weights to the given endWeights,
     * over startTime to endTime.
     */
    function updateWeightsGradually(
        uint256 startTime,
        uint256 endTime,
        uint256[] memory endWeights
    ) external authenticate whenNotPaused nonReentrant {
        InputHelpers.ensureInputLengthMatch(_getTotalTokens(), endWeights.length);

        // If the start time is in the past, "fast forward" to start now
        // This avoids discontinuities in the weight curve. Otherwise, if you set the start/end times with
        // only 10% of the period in the future, the weights would immediately jump 90%
        uint256 currentTime = block.timestamp;
        startTime = Math.max(currentTime, startTime);

        _require(startTime <= endTime, Errors.GRADUAL_UPDATE_TIME_TRAVEL);

        (IERC20[] memory tokens, , ) = getVault().getPoolTokens(getPoolId());

        _startGradualWeightChange(startTime, endTime, _getNormalizedWeights(), endWeights, tokens);
    }

    /**
     * @dev Adds an address to the allowlist.
     */
    function addAllowedAddress(address member) external authenticate whenNotPaused {
        _require(getMustAllowlistLPs(), Errors.UNAUTHORIZED_OPERATION);
        _require(!_allowedAddresses[member], Errors.ADDRESS_ALREADY_ALLOWLISTED);

        _allowedAddresses[member] = true;
        emit AllowlistAddressAdded(member);
    }

    /**
     * @dev Removes an address from the allowlist.
     */
    function removeAllowedAddress(address member) external authenticate whenNotPaused {
        _require(_allowedAddresses[member], Errors.ADDRESS_NOT_ALLOWLISTED);

        delete _allowedAddresses[member];
        emit AllowlistAddressRemoved(member);
    }

    /**
     * @dev Can enable/disable the LP allowlist. Note that any addresses added to the allowlist
     * will be retained if the allowlist is toggled off and back on again.
     */
    function setMustAllowlistLPs(bool mustAllowlistLPs) external authenticate whenNotPaused {
        _setMustAllowlistLPs(mustAllowlistLPs);
    }

    function _setMustAllowlistLPs(bool mustAllowlistLPs) private {
        _setMiscData(_getMiscData().insertBool(mustAllowlistLPs, _MUST_ALLOWLIST_LPS_OFFSET));

        emit MustAllowlistLPsSet(mustAllowlistLPs);
    }

    /**
     * @dev Enable/disable trading
     */
    function setSwapEnabled(bool swapEnabled) external authenticate whenNotPaused {
        _setSwapEnabled(swapEnabled);
    }

    function _setSwapEnabled(bool swapEnabled) private {
        _setMiscData(_getMiscData().insertBool(swapEnabled, _SWAP_ENABLED_OFFSET));

        emit SwapEnabledSet(swapEnabled);
    }

    /**
     * @dev This function takes the token, and the normalizedWeight it should have in the pool after being added.
     * The stored (denormalized) weights of all other tokens remain unchanged, but the weightSum will increase,
     * such that the normalized weight of the new token will match the target value, and the normalized weights of
     * all other tokens will be reduced proportionately.
     *
     * addToken performs the following operations:
     *
     * - Verify there is room for the token, there is no weight change, and the final weights are all valid
     * - Calculate the new BPT price, and ensure it is >= the minimum
     * - Register the new token with the Vault
     * - Join the pool, pulling in the tokenAmountIn from the sender
     * - Increase the total weight, and update the number of tokens and `_tokenState`
     *   (all other weights then scale accordingly)
     * - Return the BPT value of the new token, for possible use by the caller
     */
    function addToken(
        IERC20 token,
        uint256 normalizedWeight,
        uint256 tokenAmountIn,
        address assetManager,
        uint256 minBptPrice,
        address sender,
        address recipient
    ) external authenticate whenNotPaused returns (uint256) {
        return _addToken(token, normalizedWeight, tokenAmountIn, assetManager, minBptPrice, sender, recipient);
    }

    function _validateAddToken(
        IERC20 token,
        uint256 normalizedWeight,
        uint256 tokenAmountIn,
        uint256 minBptPrice
    ) private view returns (uint256 weightSumAfterAdd, uint256 bptAmountOut) {
        _require(normalizedWeight >= WeightedMath._MIN_WEIGHT, Errors.MIN_WEIGHT);
        // The max weight actually depends on the number of other tokens, but this is handled below
        // by ensuring the final weights are all >= minimum
        _require(normalizedWeight < FixedPoint.ONE, Errors.MAX_WEIGHT);
        _require(_getTotalTokens() < _getMaxTokens(), Errors.MAX_TOKENS);

        // Do not allow adding tokens if there is an ongoing or pending gradual weight change
        uint256 currentTime = block.timestamp;
        bytes32 poolState = _getMiscData();

        if (currentTime < poolState.decodeUint32(_START_TIME_OFFSET)) {
            _revert(Errors.CHANGE_TOKENS_PENDING_WEIGHT_CHANGE);
        } else if (currentTime < poolState.decodeUint32(_END_TIME_OFFSET)) {
            _revert(Errors.CHANGE_TOKENS_DURING_WEIGHT_CHANGE);
        }

        uint256 weightSumBeforeAdd = _denormWeightSum;

        // Calculate the weightSum after the add
        // Consider an 80/20 pool:
        // |----0.8----|-0.2-|
        // 0                 x = 1.0 (rightmost point at the "end" of the number line)
        //
        // Now add a new token with a weight of 60%
        // |----0.8----|-0.2-|---0.6y---|
        // 0                 x          y = the new weightSum
        //
        // By definition, since we interpret the new token weight as the desired final normalized weight,
        // the new "length" is the old length, plus 60% of the total new length, y
        // y = 0.6y + x
        // (1 - 0.6)y = x
        // y = x / (1 - 0.6); since x = 1, y = 1/0.4 = 2.5
        //
        // The denormalized weight of the new token, 0.6y, is then 0.6(2.5) = 1.5
        // Since the denorm weights of the original tokens stay the same, the final state is:
        // |----0.8----|-0.2-|---1.5---|  denormalized weights (as stored)
        //    0.8/2.5  0.2/2.5 1.5/2.5
        //      32%      8%      60%      normalized weights (as calculated: W[i]/weightSum)
        //
        // The added token is 60%, as desired, and the original 80/20 weights are scaled down
        // proportionately to 32/8, to fit within the remaining 40%
        //
        //uint256 weightSumMultiplier = FixedPoint.ONE.divDown(FixedPoint.ONE - normalizedWeight);
        weightSumAfterAdd = weightSumBeforeAdd.mulUp(FixedPoint.ONE.divDown(FixedPoint.ONE - normalizedWeight));

        // Now we need to make sure the other token weights don't get scaled down below the minimum
        // normalized weight[i] = denormalized weight[i] / weightSumAfterAdd
        (IERC20[] memory tokens, , ) = getVault().getPoolTokens(getPoolId());
        for (uint256 i = 0; i < tokens.length; i++) {
            _require(
                _getTokenData(tokens[i]).decodeUint64(_END_DENORM_WEIGHT_OFFSET).uncompress64(_MAX_DENORM_WEIGHT).divUp(
                    weightSumAfterAdd
                ) >= WeightedMath._MIN_WEIGHT,
                Errors.MIN_WEIGHT
            );
        }

        // Calculate the bptAmountOut equivalent to adding the token at the given weight
        // The BPT price of a token = S / (B/Wn) = (S * Wn) / B;
        // where S = totalSupply, Wn = normalized weight, and B = balance
        //
        // Since the BPT prices of existing tokens should stay constant when adding a token,
        // Let Sb, Wnb = total supply and normalized weight before the add; and
        //     Sa, Wna = total supply and normalized weight after the add
        // Then: (Sa * Wna) / B = (Sb * Wnb) / B
        //       Sa * Wna = Sb * Wnb
        //       Sa = Sb * (Wnb / Wna)
        // Since the normalized weights (Wn) = denormalized weights (Wd) / weightSum (WS), we have:
        // Sa = Sb * (Wd / WSb)  - denormalized weights don't change, so there is just one Wd
        //            ---------
        //           (Wd / WSa)
        // Sa = Sb * WSa / WSb = totalSupply after the add
        //
        // Then the bptAmountOut is the delta in the total supply:
        // bptAmountOut = Sa - Sb
        //              = Sb * WSa / WSb - Sb
        //              = Sb * (WSa / WSb - 1)
        //
        // In our example, Sa is the totalSupply on initialization of the pool. If the balances are 400 and 0.5:
        // Sb = (400^0.8) * (0.5^0.2) * 2 = 210.1222
        // Sa = 210.1222 * (2.5 / 1.0) = 525.3056
        // bptAmountOut = 210.1222 * (2.5 / 1.0 - 1) = 315.1833
        // The added token should represent 60% of the total supply, after the add, and in fact:
        // 315.1833 / 525.3056 = ~ 0.6
        //
        uint256 weightSumRatio = weightSumAfterAdd.divDown(weightSumBeforeAdd);
        bptAmountOut = totalSupply().mulDown(weightSumRatio.sub(FixedPoint.ONE));

        // Validate that the actual BPT price
        uint256 actualBptPrice = totalSupply().mulDown(weightSumRatio).mulDown(normalizedWeight).divUp(
            _upscale(tokenAmountIn, _computeScalingFactor(token))
        );

        _require(actualBptPrice >= minBptPrice, Errors.MIN_BPT_PRICE_ADD_TOKEN);
    }

    function _updateTokenStateAfterAdd(
        uint256 numTokens,
        IERC20 token,
        uint256 denormalizedWeight
    ) private {
        // Store token data, and update the token count
        bytes32 tokenState;

        _tokenState[token] = tokenState
            .insertUint64(denormalizedWeight.compress64(_MAX_DENORM_WEIGHT), _START_DENORM_WEIGHT_OFFSET)
            .insertUint64(denormalizedWeight.compress64(_MAX_DENORM_WEIGHT), _END_DENORM_WEIGHT_OFFSET)
            .insertUint5(uint256(18).sub(ERC20(address(token)).decimals()), _DECIMAL_DIFF_OFFSET);
        _setMiscData(_getMiscData().insertUint7(numTokens, _TOTAL_TOKENS_OFFSET));
    }

    function _registerNewToken(
        IERC20 token,
        uint256 denormalizedWeight,
        uint256 tokenAmountIn,
        address assetManager
    )
        private
        returns (
            IERC20[] memory,
            uint256,
            uint256[] memory
        )
    {
        address[] memory assetManagers = new address[](1);
        assetManagers[0] = assetManager;
        IERC20[] memory tokensToAdd = new IERC20[](1);
        tokensToAdd[0] = token;

        getVault().registerTokens(getPoolId(), tokensToAdd, assetManagers);

        // Tokens array is now different
        (IERC20[] memory tokens, , ) = getVault().getPoolTokens(getPoolId());
        uint256[] memory maxAmountsIn = new uint256[](tokens.length);
        uint256 tokenIndex;

        // It will put it at the end, so iterate backwards to find it faster
        for (uint256 i = tokens.length - 1; i >= 0; i--) {
            if (tokens[i] == token) {
                // This is the new one we're joining with
                maxAmountsIn[i] = tokenAmountIn;
                tokenIndex = i;
                break;
            }
        }

        _updateTokenStateAfterAdd(tokens.length, token, denormalizedWeight);

        return (tokens, tokenIndex, maxAmountsIn);
    }

    // Not worrying about updating the invariant yet, or paying protocol fees, since that will all likely change
    function _joinAddToken(
        IERC20[] memory tokens,
        uint256 tokenIndex,
        uint256 tokenAmountIn,
        uint256[] memory maxAmountsIn,
        address recipient
    ) private {
        getVault().joinPool(
            getPoolId(),
            address(this),
            payable(recipient),
            IVault.JoinPoolRequest({
                assets: _asIAsset(tokens),
                maxAmountsIn: maxAmountsIn,
                userData: abi.encode(WeightedPoolUserData.JoinKind.ADD_TOKEN, tokenIndex, tokenAmountIn),
                fromInternalBalance: false
            })
        );
    }

    /**
     * @dev 1) Validate the operation (and calculate the new weightSum and bptAmountOut to return)
     *         - the incoming normalizedWeight is valid
     *         - adding a token will not exceed the token limit
     *         - there is no ongoing or pending weight change
     *         - adding the new token at the given weight does not lower any other weights below the minimum
     *         - the final BPT price is at or above the calculated minimum
     *      2) Register the new token, with the given asset manager (and return the final sorted token list,
     *         and index of the new token). Note that the token order can completely change. Note that after
     *         registration and before joining, the pool is in an invalid state, with a zero invariant.
     *         - register the token with the Vault
     *         - adjust the management fees data structure to the new token order (preserving any uncollected fees)
     *         - adjust the rest of the token state (including token count)
     *      3) Join the pool, transferring tokens to the Vault, and restoring the pool to functional status
     *      4) Finally, update the stored weightSum, and return the bptAmountOut. The caller may then mint BPT,
     *         depending on the use case.
     */

    function _addToken(
        IERC20 token,
        uint256 normalizedWeight,
        uint256 tokenAmountIn,
        address assetManager,
        uint256 minBptPrice,
        address sender,
        address recipient
    ) internal returns (uint256) {
        (uint256 weightSumAfterAdd, uint256 bptAmountOut) = _validateAddToken(
            token,
            normalizedWeight,
            tokenAmountIn,
            minBptPrice
        );

        (IERC20[] memory tokens, uint256 tokenIndex, uint256[] memory maxAmountsIn) = _registerNewToken(
            token,
            normalizedWeight.mulUp(weightSumAfterAdd),
            tokenAmountIn,
            assetManager
        );

        // Transfer tokens from the sender to this contract, since the sender for the join must be the pool
        token.transferFrom(sender, address(this), tokenAmountIn);
        token.approve(address(getVault()), tokenAmountIn);

        _joinAddToken(tokens, tokenIndex, tokenAmountIn, maxAmountsIn, recipient);

        // If done in two stages, the controller would externally calculate a minimum BPT price (i.e., 1 token = x BPT),
        // based on dollar values.
        //
        // BPT price = (totalSupply * weight)/balance, where balance should be set to:
        // (old USD value of pool) * WSa/WSb * weight of new token
        //
        // For instance, if adding 60% DAI to our example pool with $10k of value (at $1/DAI), you would add
        // 10k * 2.5/1.0 * 0.6 = 15,000 DAI
        // The BPT price would be 525.3056 * 0.6 / 15000 = 0.021, and the controller could set a minimum of 0.02
        // (lower BPT price = higher maxAmountIn).
        //
        // In the commit stage, the actual desired balance would be passed in, and addToken would verify
        // the final BPT price.
        //
        // The controller might also impose other limitations, such as not allowing (or allowlisting) asset managers.

        _denormWeightSum = weightSumAfterAdd;

        emit TokenAdded(token, normalizedWeight, tokenAmountIn);

        return bptAmountOut;
    }

    /**
     * @dev Getter for the sum of all weights. In initially FixedPoint.ONE, it can be higher or lower
     * as a result of adds and removes.
     */
    function getDenormWeightSum() public view returns (uint256) {
        return _denormWeightSum;
    }

    /**
     * @dev Set the management fee percentage
     */
    function setManagementSwapFeePercentage(uint256 managementFeePercentage) external authenticate whenNotPaused {
        _setManagementSwapFeePercentage(managementFeePercentage);
    }

    function _setManagementSwapFeePercentage(uint256 managementSwapFeePercentage) private {
        _require(
            managementSwapFeePercentage <= _MAX_MANAGEMENT_SWAP_FEE_PERCENTAGE,
            Errors.MAX_MANAGEMENT_SWAP_FEE_PERCENTAGE
        );

        _managementSwapFeePercentage = managementSwapFeePercentage;
        emit ManagementFeePercentageChanged(managementSwapFeePercentage);
    }

    function _scalingFactor(IERC20 token) internal view virtual override returns (uint256) {
        return _readScalingFactor(_getTokenData(token));
    }

    function _scalingFactors() internal view virtual override returns (uint256[] memory scalingFactors) {
        (IERC20[] memory tokens, , ) = getVault().getPoolTokens(getPoolId());
        uint256 numTokens = tokens.length;

        scalingFactors = new uint256[](numTokens);

        for (uint256 i = 0; i < numTokens; i++) {
            scalingFactors[i] = _readScalingFactor(_tokenState[tokens[i]]);
        }
    }

    function _getNormalizedWeight(IERC20 token) internal view override returns (uint256) {
        uint256 pctProgress = _calculateWeightChangeProgress();
        bytes32 tokenData = _getTokenData(token);

        return _interpolateWeight(tokenData, pctProgress);
    }

    function _getNormalizedWeights() internal view override returns (uint256[] memory normalizedWeights) {
        (IERC20[] memory tokens, , ) = getVault().getPoolTokens(getPoolId());
        uint256 numTokens = tokens.length;

        normalizedWeights = new uint256[](numTokens);

        uint256 pctProgress = _calculateWeightChangeProgress();

        for (uint256 i = 0; i < numTokens; i++) {
            bytes32 tokenData = _tokenState[tokens[i]];

            normalizedWeights[i] = _interpolateWeight(tokenData, pctProgress);
        }
    }

    // Swap overrides - revert unless swaps are enabled

    function _onSwapGivenIn(
        SwapRequest memory swapRequest,
        uint256 currentBalanceTokenIn,
        uint256 currentBalanceTokenOut
    ) internal virtual override returns (uint256) {
        _require(getSwapEnabled(), Errors.SWAPS_DISABLED);

        (uint256[] memory normalizedWeights, uint256[] memory preSwapBalances) = _getWeightsAndPreSwapBalances(
            swapRequest,
            currentBalanceTokenIn,
            currentBalanceTokenOut
        );

        // balances (and swapRequest.amount) are already upscaled by BaseMinimalSwapInfoPool.onSwap
        uint256 amountOut = super._onSwapGivenIn(swapRequest, currentBalanceTokenIn, currentBalanceTokenOut);

        uint256[] memory postSwapBalances = ArrayHelpers.arrayFill(
            currentBalanceTokenIn.add(_addSwapFeeAmount(swapRequest.amount)),
            currentBalanceTokenOut.sub(amountOut)
        );

        _payProtocolAndManagementFees(normalizedWeights, preSwapBalances, postSwapBalances);

        return amountOut;
    }

    function _onSwapGivenOut(
        SwapRequest memory swapRequest,
        uint256 currentBalanceTokenIn,
        uint256 currentBalanceTokenOut
    ) internal virtual override returns (uint256) {
        _require(getSwapEnabled(), Errors.SWAPS_DISABLED);

        (uint256[] memory normalizedWeights, uint256[] memory preSwapBalances) = _getWeightsAndPreSwapBalances(
            swapRequest,
            currentBalanceTokenIn,
            currentBalanceTokenOut
        );

        // balances (and swapRequest.amount) are already upscaled by BaseMinimalSwapInfoPool.onSwap
        uint256 amountIn = super._onSwapGivenOut(swapRequest, currentBalanceTokenIn, currentBalanceTokenOut);

        uint256[] memory postSwapBalances = ArrayHelpers.arrayFill(
            currentBalanceTokenIn.add(_addSwapFeeAmount(amountIn)),
            currentBalanceTokenOut.sub(swapRequest.amount)
        );

        _payProtocolAndManagementFees(normalizedWeights, preSwapBalances, postSwapBalances);

        return amountIn;
    }

    function _getWeightsAndPreSwapBalances(
        SwapRequest memory swapRequest,
        uint256 currentBalanceTokenIn,
        uint256 currentBalanceTokenOut
    ) private view returns (uint256[] memory, uint256[] memory) {
        uint256[] memory normalizedWeights = ArrayHelpers.arrayFill(
            _getNormalizedWeight(swapRequest.tokenIn),
            _getNormalizedWeight(swapRequest.tokenOut)
        );

        uint256[] memory preSwapBalances = ArrayHelpers.arrayFill(currentBalanceTokenIn, currentBalanceTokenOut);

        return (normalizedWeights, preSwapBalances);
    }

    function _payProtocolAndManagementFees(
        uint256[] memory normalizedWeights,
        uint256[] memory preSwapBalances,
        uint256[] memory postSwapBalances
    ) private {
        // Calculate total BPT for the protocol and management fee
        // The management fee percentage applies to the remainder,
        // after the protocol fee has been collected.
        // So totalFee = protocolFee + (1 - protocolFee) * managementFee
        uint256 protocolSwapFeePercentage = getProtocolSwapFeePercentageCache();
        uint256 managementSwapFeePercentage = _managementSwapFeePercentage;

        if (protocolSwapFeePercentage == 0 && managementSwapFeePercentage == 0) {
            return;
        }

        // Fees are bounded, so we don't need checked math
        uint256 totalFeePercentage = protocolSwapFeePercentage +
            (FixedPoint.ONE - protocolSwapFeePercentage).mulDown(managementSwapFeePercentage);

        // No other balances are changing, so the other terms in the invariant will cancel out
        // when computing the ratio. So this partial invariant calculation is sufficient
        uint256 totalBptAmount = WeightedMath._calcDueProtocolSwapFeeBptAmount(
            totalSupply(),
            WeightedMath._calculateInvariant(normalizedWeights, preSwapBalances),
            WeightedMath._calculateInvariant(normalizedWeights, postSwapBalances),
            totalFeePercentage
        );

        // Calculate the portion of the total fee due the protocol
        // If the protocol fee were 30% and the manager fee 10%, the protocol would take 30% first.
        // Then the manager would take 10% of the remaining 70% (that is, 7%), for a total fee of 37%
        // The protocol would then earn 0.3/0.37 ~=81% of the total fee,
        // and the manager would get 0.1/0.75 ~=13%.
        uint256 protocolBptAmount = totalBptAmount.mulUp(protocolSwapFeePercentage.divUp(totalFeePercentage));

        if (protocolBptAmount > 0) {
            _payProtocolFees(protocolBptAmount);
        }

        // Pay the remainder in management fees
        // This goes to the controller, which needs to be able to withdraw them
        if (managementSwapFeePercentage > 0) {
            _mintPoolTokens(getOwner(), totalBptAmount.sub(protocolBptAmount));
        }
    }

    // We override _onJoinPool and _onExitPool as we need to not compute the current invariant and calculate protocol
    // fees, since that mechanism does not work for Pools in which the weights change over time. Instead, this Pool
    // always pays zero protocol fees.
    // Additionally, we also check that only non-swap join and exit kinds are allowed while swaps are disabled.

    function _onJoinPool(
        bytes32 poolId,
        address sender,
        address recipient,
        uint256[] memory balances,
        uint256 lastChangeBlock,
        uint256 protocolSwapFeePercentage,
        uint256[] memory scalingFactors,
        bytes memory userData
    )
        internal
        virtual
        override
        whenNotPaused // All joins are disabled while the contract is paused.
        returns (uint256 bptAmountOut, uint256[] memory amountsIn)
    {
        // If swaps are disabled, the only regular join kind that is allowed is the proportional one,
        // as all others involve implicit swaps and alter token prices. Add token is also allowed,
        // since it does not change prices of the existing tokens.
        WeightedPoolUserData.JoinKind kind = userData.joinKind();

        _require(
            getSwapEnabled() ||
                kind == WeightedPoolUserData.JoinKind.ALL_TOKENS_IN_FOR_EXACT_BPT_OUT ||
                kind == WeightedPoolUserData.JoinKind.ADD_TOKEN,
            Errors.INVALID_JOIN_EXIT_KIND_WHILE_SWAPS_DISABLED
        );

        if (WeightedPoolUserData.JoinKind.ADD_TOKEN == kind) {
            (bptAmountOut, amountsIn) = _joinAddToken(sender, scalingFactors, userData);
        } else {
            // Check allowlist for LPs, if applicable
            _require(isAllowedAddress(sender), Errors.ADDRESS_NOT_ALLOWLISTED);

            (bptAmountOut, amountsIn) = super._onJoinPool(
                poolId,
                sender,
                recipient,
                balances,
                lastChangeBlock,
                protocolSwapFeePercentage,
                scalingFactors,
                userData
            );
        }
    }

    function _joinAddToken(
        address sender,
        uint256[] memory scalingFactors,
        bytes memory userData
    ) private view returns (uint256, uint256[] memory) {
        // This join function can only be called by the Pool itself - the authorization logic that governs when that
        // call can be made resides in addToken.
        _require(sender == address(this), Errors.UNAUTHORIZED_JOIN);

        // No BPT will be issued for the join operation itself. The `addToken` function calculates and returns
        // the bptAmountOut, but leaves any further action, such as minting BPT, up to the caller.
        uint256 bptAmountOut = 0;

        // Note that there is no maximum amountsIn parameter: this is handled by `IVault.joinPool`.

        (uint256 tokenIndex, uint256 amountIn) = userData.addToken();

        uint256[] memory amountsIn = new uint256[](_getTotalTokens());
        amountsIn[tokenIndex] = _upscale(amountIn, scalingFactors[tokenIndex]);

        return (bptAmountOut, amountsIn);
    }

    function _onExitPool(
        bytes32 poolId,
        address sender,
        address recipient,
        uint256[] memory balances,
        uint256 lastChangeBlock,
        uint256 protocolSwapFeePercentage,
        uint256[] memory scalingFactors,
        bytes memory userData
    ) internal virtual override returns (uint256, uint256[] memory) {
        // Exits are not completely disabled while the contract is paused: proportional exits (exact BPT in for tokens
        // out) remain functional.

        // If swaps are disabled, the only exit kind that is allowed is the proportional one (as all others involve
        // implicit swaps and alter token prices).
        _require(
            getSwapEnabled() || userData.exitKind() == WeightedPoolUserData.ExitKind.EXACT_BPT_IN_FOR_TOKENS_OUT,
            Errors.INVALID_JOIN_EXIT_KIND_WHILE_SWAPS_DISABLED
        );

        return
            super._onExitPool(
                poolId,
                sender,
                recipient,
                balances,
                lastChangeBlock,
                protocolSwapFeePercentage,
                scalingFactors,
                userData
            );
    }

    function _tokenAddressToIndex(IERC20 token) internal view returns (uint256) {
        (IERC20[] memory tokens, , ) = getVault().getPoolTokens(getPoolId());

        for (uint256 i = 0; i < tokens.length; i++) {
            if (tokens[i] == token) {
                return i;
            }
        }

        _revert(Errors.INVALID_TOKEN);
    }

    /**
     * @dev When calling updateWeightsGradually again during an update, reset the start weights to the current weights,
     * if necessary.
     */
    function _startGradualWeightChange(
        uint256 startTime,
        uint256 endTime,
        uint256[] memory startWeights,
        uint256[] memory endWeights,
        IERC20[] memory tokens
    ) internal virtual {
        uint256 normalizedSum;

        for (uint256 i = 0; i < endWeights.length; i++) {
            uint256 endWeight = endWeights[i];
            _require(endWeight >= WeightedMath._MIN_WEIGHT, Errors.MIN_WEIGHT);
            normalizedSum = normalizedSum.add(endWeight);

            IERC20 token = tokens[i];
            _tokenState[token] = _encodeTokenState(token, startWeights[i], endWeight);
        }

        // Ensure that the normalized weights sum to ONE
        _require(normalizedSum == FixedPoint.ONE, Errors.NORMALIZED_WEIGHT_INVARIANT);

        _setMiscData(
            _getMiscData().insertUint32(startTime, _START_TIME_OFFSET).insertUint32(endTime, _END_TIME_OFFSET)
        );

        emit GradualWeightUpdateScheduled(startTime, endTime, startWeights, endWeights);
    }

    // Factored out to avoid stack issues
    function _encodeTokenState(
        IERC20 token,
        uint256 startWeight,
        uint256 endWeight
    ) private view returns (bytes32) {
        bytes32 tokenState;

        // Tokens with more than 18 decimals are not supported
        // Scaling calculations must be exact/lossless
        // Store decimal difference instead of actual scaling factor
        return
            tokenState
                .insertUint64(
                _denormalizeWeight(startWeight).compress64(_MAX_DENORM_WEIGHT),
                _START_DENORM_WEIGHT_OFFSET
            )
                .insertUint64(_denormalizeWeight(endWeight).compress64(_MAX_DENORM_WEIGHT), _END_DENORM_WEIGHT_OFFSET)
                .insertUint5(uint256(18).sub(ERC20(address(token)).decimals()), _DECIMAL_DIFF_OFFSET);
    }

    // Convert a decimal difference value to the scaling factor
    function _readScalingFactor(bytes32 tokenState) private pure returns (uint256) {
        uint256 decimalsDifference = tokenState.decodeUint5(_DECIMAL_DIFF_OFFSET);

        return FixedPoint.ONE * 10**decimalsDifference;
    }

    /**
     * @dev Extend ownerOnly functions to include the Managed Pool control functions.
     */
    function _isOwnerOnlyAction(bytes32 actionId) internal view override returns (bool) {
        return
            (actionId == getActionId(ManagedPool.updateWeightsGradually.selector)) ||
            (actionId == getActionId(ManagedPool.setSwapEnabled.selector)) ||
            (actionId == getActionId(ManagedPool.addAllowedAddress.selector)) ||
            (actionId == getActionId(ManagedPool.removeAllowedAddress.selector)) ||
            (actionId == getActionId(ManagedPool.setMustAllowlistLPs.selector)) ||
            (actionId == getActionId(ManagedPool.setManagementSwapFeePercentage.selector)) ||
            (actionId == getActionId(ManagedPool.addToken.selector)) ||
            super._isOwnerOnlyAction(actionId);
    }

    /**
     * @dev Returns a fixed-point number representing how far along the current weight change is, where 0 means the
     * change has not yet started, and FixedPoint.ONE means it has fully completed.
     */
    function _calculateWeightChangeProgress() private view returns (uint256) {
        uint256 currentTime = block.timestamp;
        bytes32 poolState = _getMiscData();

        uint256 startTime = poolState.decodeUint32(_START_TIME_OFFSET);
        uint256 endTime = poolState.decodeUint32(_END_TIME_OFFSET);

        if (currentTime >= endTime) {
            return FixedPoint.ONE;
        } else if (currentTime <= startTime) {
            return 0;
        }

        uint256 totalSeconds = endTime - startTime;
        uint256 secondsElapsed = currentTime - startTime;

        // In the degenerate case of a zero duration change, consider it completed (and avoid division by zero)
        return secondsElapsed.divDown(totalSeconds);
    }

    function _interpolateWeight(bytes32 tokenData, uint256 pctProgress) private view returns (uint256) {
        uint256 startWeight = _normalizeWeight(
            tokenData.decodeUint64(_START_DENORM_WEIGHT_OFFSET).uncompress64(_MAX_DENORM_WEIGHT)
        );
        uint256 endWeight = _normalizeWeight(
            tokenData.decodeUint64(_END_DENORM_WEIGHT_OFFSET).uncompress64(_MAX_DENORM_WEIGHT)
        );

        if (pctProgress == 0 || startWeight == endWeight) return startWeight;
        if (pctProgress >= FixedPoint.ONE) return endWeight;

        if (startWeight > endWeight) {
            uint256 weightDelta = pctProgress.mulDown(startWeight - endWeight);
            return startWeight - weightDelta;
        } else {
            uint256 weightDelta = pctProgress.mulDown(endWeight - startWeight);
            return startWeight + weightDelta;
        }
    }

    function _getTokenData(IERC20 token) private view returns (bytes32 tokenData) {
        tokenData = _tokenState[token];

        // A valid token can't be zero (must have non-zero weights)
        _require(tokenData != 0, Errors.INVALID_TOKEN);
    }

    // Functions that convert weights between internal (denormalized) and external (normalized) representations

    // Convert from the internal representation to normalized weights (summing to ONE)
    function _normalizeWeight(uint256 denormWeight) private view returns (uint256) {
        return denormWeight.divDown(_denormWeightSum);
    }

    // converts from normalized form to the internal representation (summing to _denormWeightSum)
    function _denormalizeWeight(uint256 weight) private view returns (uint256) {
        return weight.mulUp(_denormWeightSum);
    }
}
