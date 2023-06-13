// 200. Number of Islands - medium
/**
 * @param {character[][]} grid
 * @return {number}
 */
var numIslands = function (grid) {
    let ans = 0;
    let m = grid.length;
    let n = grid[0].length;
    let dir = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (grid[i][j] === '1') {
                ans++;
                grid[i][j] = '0';
                let q = [[i, j]];
                while (q.length) {
                    p = q.shift();
                    for (let d of dir) {
                        let x = p[0] + d[0];
                        let y = p[1] + d[1];
                        if (0 <= x && x < m && 0 <= y && y < n && grid[x][y] === '1') {
                            grid[x][y] = '0';
                            q.push([x, y]);
                        }
                    }
                }
            }
        }
    }
    return ans;
};

var numIslands = function (grid) {
    let ans = 0;
    let m = grid.length;
    let n = grid[0].length;
    function dfs(grid, r, c) {
        if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] === '0') return;
        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (grid[i][j] === '1') {
                ans++;
                dfs(grid, i, j);
            }
        }
    }
    return ans;
};