/*

模版:
    并查集
    二分
    堆 TODO
    数组相关


技巧:
    Math.floor(x) == x >> 1
    [x, y] = [y, x]

*/


/**
 * 并查集
 */

class UnionFind {
    constructor(n) {
        this.p = new Array(n).fill(0).map((v, index) => index);
        this.sz = new Array(n).fill(1);
        this.part = n;
    }
    find(x) {
        if (this.p[x] === x) {
            return x;
        }
        this.p[x] = this.find(this.p[x]);
        return this.p[x];
    }
    union(x, y) {
        let px = this.find(x), py = this.find(y);
        if (px === py) {
            return false;
        }
        if (this.sz[px] < this.sz[py]) {
            [px, py] = [py, px];
        }
        this.p[py] = px;
        this.sz[px] += this.sz[py];
        this.part -= 1;
        return true;
    }
    connected(x, y) {
        const px = this.find(x), py = this.find(y);
        return px === py;
    }
}

class UnionFind {
    constructor(num) { // num为顶点个数
        this.roots = new Array(num)     // 初始化roots数组
        for (let i = 0; i < num; i++) { // 元素初始化为-1
            this.roots[i] = -1
        }
    }
    findRoot(x) { // 找出顶点x的根节点
        let x_root = x // 先从x节点开始
        while (this.roots[x_root] !== -1) { // 一直找父节点，找到尽头
            x_root = this.roots[x_root]
        }
        return x_root // 返回根节点
    }
    union(x, y) { // 把顶点x和顶点y所在的集合合并到一起
        let x_root = this.findRoot(x)
        let y_root = this.findRoot(y) // x, y 各自的根节点
        if (x_root === y_root) return // 如果根节点相同，说明已经在一个集合，直接返回
        roots[x_root] = y_root   // 让x的根节点指向y的根节点，就合并了两个树
    }
}

class UnionFind {
    constructor(num) { // num 顶点个数
        this.roots = new Array(num)
        this.ranks = new Array(num)
        for (let i = 0; i < num; i++) {
            this.roots[i] = -1
            this.ranks[i] = 0
        }
    }
    findRoot(x) { } // 代码同上，省略
    union(x, y) { // 把顶点x和顶点y所在的集合合并到一起
        let x_root = this.findRoot(x)
        let y_root = this.findRoot(y)
        if (x_root === y_root) return  // 已经同处于一个集合了
        let x_rank = this.ranks[x_root]
        let y_rank = this.ranks[y_root]
        if (x_rank < y_rank) {    // 谁高度大，谁就作为根节点
            this.roots[x_root] = y_root
        } else if (y_rank < x_rank) {
            this.roots[y_root] = x_root
        } else {                  // 一样高，谁作为根节点都行
            this.roots[y_root] = x_root
            this.ranks[x_root]++    // 作为根节点的，高度会+1
        }
    }
}

/**
 * 二分
 */

// lower_bound -> bisect_left
function lower_bound(arr, x) {
    let l = 0, r = arr.length;
    while (l < r) {
        let m = l + Math.floor((r - l) / 2);
        if (arr[m] >= x) r = m;
        else l = m + 1;
    }
    return l;
}

// upper_bound -> bisect_right
function upper_bound(arr, x) {
    let l = 0, r = arr.length;
    while (l < r) {
        let m = l + Math.floor((r - l) / 2);
        if (arr[m] > x) r = m;
        else l = m + 1;
    }
    return l;
}

/**
 * 数组相关
 * 
 * 二维数组初始化
 * 排序
 * 去重
 * 拷贝
 */

let arr = [];
let cp = [];
let f = [];

// 二维数组初始化
let m = 3, n = 4 // m 行 n 

f = new Array(m);
for (var i = 0; i < f.length; i++) {
    f[i] = new Array(n).fill(0);
}

for (var i = 0; i < m; i++) {
    f[i] = new Array();
    for (var j = 0; j < n; j++) {
        f[i][j] = 0;
    }
}

f = Array.from(Array(m)).map(() => Array(n).fill(0))
f = [...Array(m)].map(() => Array(n).fill(0))

// 排序

// 一维
arr.sort((x, y) => x - y) // 升序
arr.sort((x, y) => y - x) // 降序

arr.sort(asc)
function asc(x, y) { return x - y }
function desc(x, y) { return y - x }

// 二维
function asc(x, y) { return x[1] - y[1] }
function desc(x, y) { return y[1] - x[1] }

// 多列自定义
function cmp(x, y) { return x[1] - y[1] || y[0] - x[0] } // 第二列升序 -> 第一列降序

mi = Math.min.apply(null, arr)
mx = Math.max.apply(null, arr)

// 去重, 会打乱排序
function unique(arr) {
    return Array.from(new Set(arr)) // 无法去掉 "{}" 空对象
}
arr = [...new Set(arr)] // 同上

// 浅拷贝
// 如果数组里嵌套数组和对象, 浅拷贝只会拷贝该数组或者对象存放在栈空间的地址, 所以不能有效的拷贝多维数组
cp = [...arr]
cp = arr.slice()
cp = arr.concat()
arr.forEach(v => { cp.push(v) })

// 深拷贝
cp = JSON.parse(JSON.stringify(arr)) // 能拷贝数组和对象，但不能拷贝函数