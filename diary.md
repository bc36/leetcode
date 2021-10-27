* 9.13
map 拷贝
go map 不能copy，只能循环重新赋给新map
```go
targetMap := make(map[string]int)
for key := range originalMap {
    targetMap[key] = originalMap[key]
}
// or
for key, value := range originalMap {
  targetMap[key] = value
}
```
浅拷贝：值类型完全copy一份，引用类型：拷贝其地址。修改变量影响原有变量(map, slice)
深拷贝：完整copy


* 深拷贝 1.序列化 2.反射(速度比序列化快)
```go
// 基于序列化和反序列化来实现对象的深度拷贝
// gob
gob.NewEncoder(buffer).Encode(*teamA)
gob.NewDecoder(bytes.NewBuffer(buffer.Bytes())).Decode(teamB)
//
func deepCopy(dst, src interface{}) error {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(src); err != nil {
		return err
	}
	return gob.NewDecoder(bytes.NewBuffer(buf.Bytes())).Decode(dst)
}
// json 一般效率较高、结构体比较小时候
buffer, _ := json.Marshal(&teamA)
json.Unmarshal([]byte(buffer), teamB)
```
```go
// 反射
func copyRecursive(original, cpy reflect.Value) {
	switch original.Kind() {
	case reflect.Ptr:
		// 获取指针所指的实际值，并拷贝给cpy
		originalValue := original.Elem()
		cpy.Set(reflect.New(originalValue.Type()))
		copyRecursive(originalValue, cpy.Elem())

	case reflect.Struct:
		// 遍历结构体中的每一个成员
		for i := 0; i < original.NumField(); i++ {
			copyRecursive(original.Field(i), cpy.Field(i))
		}

	case reflect.Slice:
		// 在cpy中创建一个新的切片，并使用一个for循环拷贝
		cpy.Set(reflect.MakeSlice(original.Type(), original.Len(), original.Cap()))
		for i := 0; i < original.Len(); i++ {
			copyRecursive(original.Index(i), cpy.Index(i))
		}

	default:
		// 真正的拷贝操作
		cpy.Set(original)
	}
}
```
```go
// 某一个包里的样例，处理map copy
package utils

func CopyMap(m map[string]interface{}) map[string]interface{} {
    cp := make(map[string]interface{})
    for k, v := range m {
        vm, ok := v.(map[string]interface{})
        if ok {
            cp[k] = CopyMap(vm)
        } else {
            cp[k] = v
        }
    }

    return cp
}
```

* 9.22
`a = strings.Split(s, sep)` s和sep都为空时，`len(a) = 1, cap(a) = 1`
reason：空字符串包含一个空字符串

* 10.5
pid
```c++
previous_error := 0
integral := 0
loop :
	error := ideal_distance - measured_distance
	integral := integral + error * dt
	derivative := (error - previous_error) / dt
	output := Kp * error + Ki * integral + Kd * derivative
	previous_error := error
	wait(dt)
	goto loop
```

* 10.25
go slice 切片陷阱
```go
/* 
如果您想要将它与初始的切片分开请不要忘记 copy()
对于 append 函数，忘记 copy() 会变得更加危险:如果它没有足够的容量来保存新值，底层数组将会重新分配内存和大小。这意味着 append 的结果能不能指向原始数组取决于它的初始容量。这会导致难以发现的不确定 bugs。
在下面的代码中，我们看到为子切片追加值的影响取决于原始切片的容量:
*/
func doStuff(value []string) {
    fmt.Printf("value=%v\n", value)

    value2 := value[:]
    value2 = append(value2, "b")
    fmt.Printf("value=%v, value2=%v\n", value, value2)

    value2[0] = "z"
    fmt.Printf("value=%v, value2=%v\n", value, value2)
}

func main() {
    slice1 := []string{"a"} // 长度 1, 容量 1

    doStuff(slice1)
    // Output:
    // value=[a] -- ok
    // value=[a], value2=[a b] -- ok: value 未改变, value2 被更新
    // value=[a], value2=[z b] -- ok: value 未改变, value2 被更新

    slice10 := make([]string, 1, 10) // 长度 1, 容量 10
    slice10[0] = "a"

    doStuff(slice10)
    // Output:
    // value=[a] -- ok
    // value=[a], value2=[a b] -- ok: value 未改变, value2 被更新
    // value=[z], value2=[z b] -- WTF?!? value 改变了???
}

```