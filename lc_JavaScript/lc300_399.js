// 322. Coin Change - medium
/**
 * @param {number[]} coins
 * @param {number} amount
 * @return {number}
 */
var coinChange = function (coins, amount) {
    const f = new Array(amount + 1).fill(Infinity);
    f[0] = 0;
    for (let i = 0; i <= amount; i++) {
        for (let c of coins) {
            if (i < c) { continue };
            f[i] = Math.min(f[i], f[i - c] + 1);
        }
    }
    return f[amount] === Infinity ? -1 : f[amount];
};

var coinChange = function (coins, amount) {
    const f = new Array(amount + 1).fill(Infinity);
    f[0] = 0;
    for (let c of coins) {
        for ( let i = c; i <= amount; i++){
            f[i] = Math.min(f[i], f[i - c] + 1);
        }
    }
    return f[amount] === Infinity ? -1 : f[amount];
};