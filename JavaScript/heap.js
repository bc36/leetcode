// 大顶堆
/*
let q = new Heap((a, b) => a < b) // 转为小顶堆

q.pop()
q.peek()
q.push(x)
...

*/
class Heap {
    constructor(compare) {
        this.arr = [0]; // 忽略 0 这个索引,方便计算左右节点
        this.compare = compare ? compare : (a, b) => a > b; // 默认是大顶堆
    }
    get size() {
        return this.arr.length - 1;
    }
    // 新增元素
    push(x) {
        this.arr.push(x);
        this.shiftUp(this.arr.length - 1);
    }
    // 弹出堆顶
    pop() {
        if (this.arr.length == 1) return null;
        this.swap(1, this.arr.length - 1);// 将结尾元素和堆顶元素交换位置
        let head = this.arr.pop(); // 删除堆顶
        this.sinkDown(1); // 再做下沉操作
        return head;
    }
    // 元素上浮，k 是索引
    shiftUp(k) {
        let { arr, compare, parent } = this;
        // 当前元素 > 父元素，则进行交换
        while (k > 1 && compare(arr[k], arr[parent(k)])) {
            this.swap(parent(k), k);
            k = parent(k); // 更新 k 的位置为父元素的位置（交换了位置）
        }
    }
    // 元素下沉
    sinkDown(k) {
        let { arr, compare, left, right, size } = this;
        while (left(k) <= size) {
            // 1. 拿到左右节点的最大值
            let child = left(k);
            if (right(k) <= size && compare(arr[right(k)], arr[child])) {
                child = right(k);
            }
            // 2. k 满足大顶堆或小顶堆，什么都不做
            if (compare(arr[k], arr[child])) {
                return;
            }
            // 3. 交换位置后，继续下沉操作
            this.swap(k, child);
            k = child; // 更新位置
        }
    }
    // 获取堆顶元素
    peek() {
        return this.arr[1];
    }
    // 获取左边元素节点
    left(k) {
        return k * 2;
    }
    // 获取右边元素节点
    right(k) {
        return k * 2 + 1;
    }
    // 获取父节点
    parent(k) {
        return Math.floor(k >> 1);
    }
    // i、j 交换位置
    swap(i, j) {
        [this.arr[i], this.arr[j]] = [this.arr[j], this.arr[i]];
    }
}


// 小顶堆 有问题
class MinHeap {
    constructor() {
        this.heap = [];
    }
    getParentIndex(i) {
        return (i - 1) >> 1;
    }
    getLeftIndex(i) {
        return i * 2 + 1;
    }
    getRightIndex(i) {
        return i * 2 + 2;
    }
    shiftUp(index) {
        if (index === 0) { return; }
        const parentIndex = this.getParentIndex(index);
        if (this.heap[parentIndex] > this.heap[index]) {
            this.swap(parentIndex, index);
            this.shiftUp(parentIndex);
        }
    }
    swap(i1, i2) {
        const temp = this.heap[i1];
        this.heap[i1] = this.heap[i2];
        this.heap[i2] = temp;
    }
    push(v) {
        this.heap.push(v);
        this.shiftUp(this.heap.length - 1);
    }
    pop() {
        this.heap[0] = this.heap.pop();
        this.shiftDown(0);
        return this.heap[0];
    }
    shiftDown(index) {
        const leftIndex = this.getLeftIndex(index);
        const rightIndex = this.getRightIndex(index);
        if (this.heap[leftIndex] < this.heap[index]) {
            this.swap(leftIndex, index);
            this.shiftDown(leftIndex);
        }
        if (this.heap[rightIndex] < this.heap[index]) {
            this.swap(rightIndex, index);
            this.shiftDown(rightIndex);
        }
    }
    peek() {
        return this.heap[0];
    }
    size() {
        return this.heap.length;
    }
}

// 小顶堆
class Heap {
    constructor(comparator = (a, b) => a - b, data = []) {
        this.data = data;
        this.comparator = comparator;//比较器
        this.heapify();//堆化
    }
    heapify() {
        if (this.size() < 2) return;
        for (let i = Math.floor(this.size() / 2) - 1; i >= 0; i--) {
            this.bubbleDown(i);//bubbleDown操作
        }
    }
    peek() {
        if (this.size() === 0) return null;
        return this.data[0];//查看堆顶
    }
    offer(value) {
        this.data.push(value);//加入数组
        this.bubbleUp(this.size() - 1);//调整加入的元素在小顶堆中的位置
    }
    poll() {
        if (this.size() === 0) {
            return null;
        }
        const result = this.data[0];
        const last = this.data.pop();
        if (this.size() !== 0) {
            this.data[0] = last;//交换第一个元素和最后一个元素
            this.bubbleDown(0);//bubbleDown操作
        }
        return result;
    }
    bubbleUp(index) {
        while (index > 0) {
            const parentIndex = (index - 1) >> 1; //父节点的位置
            //如果当前元素比父节点的元素小，就交换当前节点和父节点的位置
            if (this.comparator(this.data[index], this.data[parentIndex]) < 0) {
                this.swap(index, parentIndex);//交换自己和父节点的位置
                index = parentIndex;//不断向上取父节点进行比较
            } else {
                break;//如果当前元素比父节点的元素大，不需要处理
            }
        }
    }
    bubbleDown(index) {
        const lastIndex = this.size() - 1; //最后一个节点的位置
        while (true) {
            const leftIndex = index * 2 + 1;//左节点的位置
            const rightIndex = index * 2 + 2;//右节点的位置
            let findIndex = index;//bubbleDown节点的位置
            //找出左右节点中value小的节点
            if (
                leftIndex <= lastIndex &&
                this.comparator(this.data[leftIndex], this.data[findIndex]) < 0
            ) {
                findIndex = leftIndex;
            }
            if (
                rightIndex <= lastIndex &&
                this.comparator(this.data[rightIndex], this.data[findIndex]) < 0
            ) {
                findIndex = rightIndex;
            }
            if (index !== findIndex) {
                this.swap(index, findIndex);//交换当前元素和左右节点中value小的
                index = findIndex;
            } else {
                break;
            }
        }
    }
    swap(index1, index2) { //交换堆中两个元素的位置
        [this.data[index1], this.data[index2]] = [this.data[index2], this.data[index1]];
    }
    size() {
        return this.data.length;
    }
}

var topKFrequent = function (nums, k) {
    const map = new Map();

    for (const num of nums) {//统计频次
        map.set(num, (map.get(num) || 0) + 1);
    }

    //创建小顶堆
    const priorityQueue = new Heap((a, b) => a[1] - b[1]);

    //entry 是一个长度为2的数组，0位置存储key，1位置存储value
    for (const entry of map.entries()) {
        priorityQueue.offer(entry);//加入堆
        if (priorityQueue.size() > k) {//堆的size超过k时，出堆
            priorityQueue.poll();
        }
    }

    const ret = [];

    for (let i = priorityQueue.size() - 1; i >= 0; i--) {//取出前k大的数
        ret[i] = priorityQueue.poll()[0];
    }

    return ret;
};


// 小顶堆 有问题
function PriorityQueue(compareFn) {
    this.compareFn = compareFn;
    this.queue = [];
}

// 添加
PriorityQueue.prototype.push = function (item) {
    this.queue.push(item);
    let index = this.queue.length - 1;
    let parent = Math.floor((index - 1) / 2);
    // 上浮
    while (parent >= 0 && this.compare(parent, index) > 0) {
        // 交换
        [this.queue[index], this.queue[parent]] = [this.queue[parent], this.queue[index]];
        index = parent;
        parent = Math.floor((index - 1) / 2);
    }
}

// 获取堆顶元素并移除
PriorityQueue.prototype.pop = function () {
    const ret = this.queue[0];

    // 把最后一个节点移到堆顶
    this.queue[0] = this.queue.pop();

    let index = 0;
    // 左子节点下标，left + 1 就是右子节点下标
    let left = 1;
    let selectedChild = this.compare(left, left + 1) > 0 ? left + 1 : left;

    // 下沉
    while (selectedChild !== undefined && this.compare(index, selectedChild) > 0) {
        // 交换
        [this.queue[index], this.queue[selectedChild]] = [this.queue[selectedChild], this.queue[index]];
        index = selectedChild;
        left = 2 * index + 1;
        selectedChild = this.compare(left, left + 1) > 0 ? left + 1 : left;
    }

    return ret;
}

PriorityQueue.prototype.size = function () {
    return this.queue.length;
}

// 使用传入的 compareFn 比较两个位置的元素
PriorityQueue.prototype.compare = function (index1, index2) {
    if (this.queue[index1] === undefined) {
        return 1;
    }
    if (this.queue[index2] === undefined) {
        return -1;
    }

    return this.compareFn(this.queue[index1], this.queue[index2]);
}