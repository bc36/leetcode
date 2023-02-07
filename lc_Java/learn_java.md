# **Java**

## **笔记**
### **Array / List**
* 与 ArrayList 相比, LinkedList 的增加和删除的操作效率更高, 而查找和修改的操作效率较低
    * 以下情况使用 ArrayList:
        * 频繁访问列表中的某一个元素.
        * 只需要在列表末尾进行添加和删除元素操作.
    * 以下情况使用 LinkedList:
        * 你需要通过循环迭代来访问列表中的某些元素.
        * 需要频繁的在列表开头, 中间, 末尾等位置进行添加和删除元素操作

### **Map / Set**
* HashSet 是无序的, HashSet 不是线程安全的, HashSet 基于 HashMap 来实现的
* 为什么申明方法一用的比较多?
    ```java
    Map<String> m = new HashMap<String>();
    HashMap<String> m = new HashMap<String>();
    // A1: 
    // 面向接口编程的体现. 如果都声明成第二种的形式, 接口还有什么用?
    // 声明成接口的形式, 后面的使用不管你类的实现, 比如说后面你发现HashMap不合适了, 你只需要修改一处就行
    // A2: 
    // 假设你一个函数中的参数是使用HashMap声明, 那当你想要修改为Hashtable时需要修改函数中的参数类型, 而使用Map就不会遇到这种问题
    // A3: 
    // 第一种声明方式是: 父类的引用指向子类的对象,是多态的一种表现形式; 第二种是: 使用本身引用
    // 第一种声明方式是实现了多态, 多态后就可以写出一段所有子类都通用的代码, 当添加新的子类时, 这段代码是不需要修改的.
    //      比方说: 父类Animal, 子类Cat, Dog
    //      如果使用第2行, 当改用Dog的时候, 那么第3行也需要改变
    //      1 Animal a = new Cat();
    //      2 Cat a = new Cat();
    //      3 a.someMethod();
    // 父类的引用指向子类的对象的好处: 多态, 动态链接, 向上转型. 方法的重写, 重载与动态连接构成多态性. 
    // Java之所以引入多态的概念, 原因之一是它在类的继承问题上和C++不同, 
    // 后者允许多继承, 这确实给其带来的非常强大的功能, 但是复杂的继承关系也给C++开发者带来了更大的麻烦, 
    // 为了规避风险, Java只允许单继承, 派生类与基类间有IS-A的关系(即“猫” is a "动物"). 
    // 这样做虽然保证了继承关系的简单明了, 但是势必在功能上有很大的限制, 
    // 所以, Java引入了多态性的概念以弥补这点的不足, 此外, 抽象类和接口也是解决单继承规定限制的重要手段. 
    // 同时, 多态也是面向对象编程的精髓所在
    ```

## **Java 基础速读**

### **语法基础**
```java
// * 声明与初始化
int x = 1 + 2;
short sh = (short) (x); // 强制转型, 即将大范围的整数转型为小范围的整数, 小转大会出错
int n = (int) 12.3; // 12

// * java数组特点:
// 1. 长度不可变
// 2. 数组所有元素初始化为默认值
int[] l = new int[5];
int[] l2 = new int[] { 1, 1, 1 };
int[] l1 = { 1, 1 };

int[] l3;
l3 = new int[] { 68, 79, 91, 85, 62 };
System.out.println(l3.length); // 5
l3 = new int[] { 1, 2, 3 };
System.out.println(l3.length); // 3

String[] names = { "ABC", "XYZ", "zoo" };
String s = names[1];
names[1] = "cat";
System.out.println(s); // s是"XYZ"还是"cat"? -> XYZ

int[] l4 = { 1, 1, 2, 3, 5, 8 };
System.out.println(l4); // 类似 [I@7852e922, 打印数组在JVM中的引用地址
System.out.println(Arrays.toString(l4)); // 标准库提供了Arrays.toString()

// * var 关键字
StringBuilder sb = new StringBuilder();
var sbb = new StringBuilder();

// * 字符和字符串
// 是两个不同的类型, char是基本数据类型, 一个char保存一个Unicode, 字符String是引用类型
char c1 = 'A';
char c2 = '中';
// Java在内存中总是使用Unicode表示字符, 所以, 一个英文字符和一个中文字符都用一个char类型表示,
// 它们都占用两个字节. 要显示一个字符的Unicode编码, 只需将char类型直接赋值给int类型即可
int n1 = 'A'; // 字母“A”的Unicodde编码是65
int n2 = '中'; // 汉字“中”的Unicode编码是20013
// 还可以直接用转义字符 \\u+Unicode 编码来表示一个字符:
int n3 = 'A'; // 字母“A”的Unicodde编码是65
int n4 = '中'; // 汉字“中”的Unicode编码是20013
// 如果用+连接字符串和其他数据类型, 会将其他数据类型先自动转型为字符串, 再连接
String s5 = "age is " + x;

String s4 = "hello";
String t = s4;
s4 = "world";
System.out.println(t); // t是"hello"还是"world"? -> hello

// * 空值null
// 引用类型的变量可以指向一个空值null, 它表示不存在, 即该变量不指向任何对象
// 注意要区分空值null和空字符串"", 空字符串是一个有效的字符串对象, 它不等于null
String s1 = null; // s1是null
String s2 = s1; // s2也是null
String s3 = ""; // s3指向空字符串, 不是null
```

### **面向对象(OOP)**
class 和 instance
定义class就是定义了一种数据类型, 对应的instance是这种数据类型的实例.
通过new操作符创建新的instance, 然后用变量指向它, 即可通过变量来引用这个instance.
一个Java源文件可以包含多个类的定义, 但只能定义一个public类, 且public类名必须与文件名一致.
如果要定义多个public类, 必须拆到多个Java源文件中.


#### **方法**

public / private, class 内部调用

this
如果没有命名冲突 可以省略this, 但是, 如果有局部变量和字段重名, 那么局部变量优先级更高, 就必须加上this. 
```java
class Person {
    private String name;

    public String getName() {
        return name; // 相当于this.name
    }
}
```
可变参数用 [类型...] 定义, 可变参数相当于数组类型
```java
public void setNames(String... names) {
    this.names = names;
}
```
完全可以把可变参数改写为String[]类型, 但是调用方需要自己先构造String[], 比较麻烦. 
`g.setNames(new String[] {"Xiao Ming", "Xiao Jun"}); // 传入1个String[]`. 
调用方可以传入null, 而可变参数可以保证无法传入null, 因为传入0个参数时, 接收到的实际值是一个空数组而不是null. 

引用类型参数的传递, 调用方的变量, 和接收方的参数变量, 指向的是同一个对象. 
双方任意一方对这个对象的修改, 都会影响对方, 类似在python中传递一个数组

#### **构造方法**

构造方法的名称就是类名. 构造方法的参数没有限制, 在方法内部, 也可以编写任意语句. 
和普通方法相比, 构造方法没有返回值(也没有void), 调用构造方法, 必须用new操作符. 

任何class都有构造方法？是的.
如果一个类没有定义构造方法, 编译器会自动为我们生成一个默认构造方法, 它没有参数, 也没有执行语句, 类似
```java
class Person {
    public Person() {
    }
}
```
如果既要能使用带参数的构造方法, 又想保留不带参数的构造方法, 那么只能把两个构造方法都定义出来(多个构造方法). 
可以定义多个构造方法, 在通过new操作符调用的时候, 编译器通过构造方法的参数数量、位置和类型自动区分
没有在构造方法中初始化字段时
* 引用类型的字段默认是null
* 数值类型的字段用默认值
* int类型默认值是0
* 布尔类型默认值是false

问: 既对字段进行初始化, 又在构造方法中对字段进行初始化, 得到的对象实例, 字段的初始值是?
1. 先初始化字段
2. 执行构造方法的代码进行初始化

所以, 由于构造方法的代码后运行, 字段值最终由构造方法的代码确定. 
一个构造方法可以调用其他构造方法, 这样做的目的是便于代码复用. 调用其他构造方法的语法是this(...)
```java
class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public Person(String name) {
        this(name, 18); // 调用另一个构造方法Person(String, int)
    }

    public Person() {
        this("Unnamed"); // 调用另一个构造方法Person(String)
    }
}
```

#### **重载**
这种方法名相同, 但各自的参数不同, 称为方法重载(Overload)(方法重载的返回值类型通常都是相同的)

#### **继承**
extends 关键字: `class 子类 extends 父类 {}`.
子类自动获得了父类的所有字段, 严禁定义与父类重名的字段!

在OOP中:  
Person称为超类(super class), 父类(parent class), 基类(base class),
Student称为子类(subclass), 扩展类(extended class).

没有明确写extends的类, 编译器会自动加上extends Object. 
Java只允许一个class继承自一个类, 因此, 一个类有且仅有一个父类. 只有Object特殊, 它没有父类

继承特点:  
子类无法访问父类的private字段或者private方法. 
为了让子类可以访问父类的字段, 我们需要把private改为protected, 用protected修饰的字段可以被子类访问
protected关键字可以把字段和方法的访问权限控制在继承树内部

super关键字表示父类(超类). 子类引用父类的字段时, 可以用`super.fieldName`
在Java中, 任何class的构造方法, 第一行语句必须是调用父类的构造方法. 如果没有明确地调用父类的构造方法, 编译器会帮我们自动加一句super().
```java
class Student extends Person {
    protected int score;

    public Student(String name, int age, int score) {
        super(name, age); // 调用父类的构造方法Person(String, int)
        this.score = score;
    }
}
```
如果父类没有默认的构造方法, 子类就必须显式调用super()并给出参数以便让编译器定位到父类的一个合适的构造方法
即子类不会继承任何父类的构造方法. 子类默认的构造方法是编译器自动生成的, 不是继承的

阻止继承:  
只要某个class没有final修饰符, 那么任何类都可以从该class继承

向上转型(upcasting):  
如果Student是从Person继承下来的, 那么, 一个引用类型为Person的变量, 能否指向Student类型的实例？  
`Person p = new Student(); // 是允许的`  
向上转型实际上是把一个子类型安全地变为更加抽象的父类型

向下转型(downcasting):  
instanceof 先判断一个实例究竟是不是某种类型, 失败的时候, Java虚拟机会报ClassCastException
如果一个引用变量为null, 那么对任何instanceof的判断都为false

具有has关系不应该使用继承, 而是使用组合, 即Student可以持有一个Book实例:
```java
class Student extends Person {
    protected Book book;
    protected int score;
}
```

#### **多态**
在继承关系中, 子类如果定义了一个与父类方法签名完全相同的方法, 被称为覆写(Override). 
Override和Overload不同的是, 如果方法签名不同, 就是Overload, Overload方法是一个新方法；如果方法签名相同, 并且返回值也相同, 就是Override. 
注意; 方法名相同, 方法参数相同, 但方法返回值不同, 也是不同的方法. 在Java程序中, 出现这种情况, 编译器会报错. 加上 `@Override` 以供检查
```java
Person p = new Student();
p.run(); // 打印Person.run还是Student.run? -> 打印 Student.run
```
Java的实例方法调用是基于运行时的实际类型的动态调用, 而非变量的声明类型
在面向对象编程中称之为多态. 它的英文拼写非常复杂; Polymorphic
多态是指, 针对某个类型的方法调用, 其真正执行的方法取决于运行时期实际类型的方法

在子类的覆写方法中, 如果要调用父类的被覆写的方法, 可以通过super来调用
```java
class Student extends Person {
    @Override
    public String hello() {
        // 调用父类的hello()方法:
        return super.hello() + "!";
    }
}
```

#### **final**
如果一个父类不允许子类对它的某个方法进行覆写, 可以把该方法标记为final. 用final修饰的方法不能被Override
```java
class Person {
    public final String hello() {
        return "Hello, " + name;
    }
}
```
如果一个类不希望任何其他类继承自它, 那么可以把这个类本身标记为final. 用final修饰的类不能被继承
```java
final class Person {
    protected String name;
}
```
用final修饰的字段在初始化后不能被修改
```java
class Person {
    public final String name = "Unamed";
}
```
但是可以在构造方法中初始化final字段
```java
class Person {
    public final String name;
    public Person(String name) {
        this.name = name;
    }
}
```

* 抽象类
如果父类的方法本身不需要实现任何功能, 仅仅是为了定义方法签名, 目的是让子类去覆写它, 那么, 可以把父类的方法声明为抽象方法
```java
class Person {
    public abstract void run(); // ERROR!
}
```

把一个方法声明为abstract, 表示它是一个抽象方法, 本身没有实现任何方法语句. 
因为这个抽象方法本身是无法执行的, 所以, Person类也无法被实例化. 编译器会告诉我们, 无法编译Person类, 因为它包含抽象方法. 
必须把Person类本身也声明为abstract, 才能正确编译它
```java
abstract class Person {
    public abstract void run();
}
```
抽象类本身被设计成只能用于被继承, 因此, 抽象类可以强迫子类实现其定义的抽象方法, 否则编译会报错. 因此, 抽象方法实际上相当于定义了“规范”. 
从抽象类继承的子类必须实现抽象方法, 面向抽象编程使得调用者只关心抽象方法的定义, 不关心子类的具体实现

#### **接口**
如果一个抽象类没有字段, 所有方法全部都是抽象方法, 就可以把该抽象类改写为接口: interface
```java
abstract class Person {
    public abstract void run();
    public abstract String getName();
}
//退化为
interface Person {
    void run();
    String getName();
}
```
所谓interface, 就是比抽象类还要抽象的纯抽象接口, 因为它连字段都不能有. 因为接口定义的所有方法默认都是public abstract的, 所以这两个修饰符不需要写出来(写不写效果都一样). 
当一个具体的class去实现一个interface时, 需要使用implements关键字.
```java
class Student implements Person {
    private String name;

    public Student(String name) {
        this.name = name;
    }

    @Override
    public void run() {
        System.out.println(this.name + " run");
    }

    @Override
    public String getName() {
        return this.name;
    }
}
// 在Java中, 一个类只能继承自另一个类, 不能从多个类继承. 但是, 一个类可以实现多个interface
class Student implements Person, Hello { // 实现了两个interface
    ...
}
```

抽象类和接口的对比如下:  
abstract    | class                 | interface
---------   | -----                 | -------
继承        | 只能extends一个class    | 可以implements多个interface
字段        | 可以定义实例字段         |不能定义实例字段
抽象方法     | 可以定义抽象方法         |可以定义抽象方法
非抽象方法   | 可以定义非抽象方法       |可以定义default方法

一个interface可以继承自另一个interface. interface继承自interface使用extends, 它相当于扩展了接口的方法.

#### **default**
实现类可以不必覆写default方法.  
default方法的目的是, 当我们需要给接口新增一个方法时, 会涉及到修改全部子类. 
如果新增的是default方法, 那么子类就不必全部修改, 只需要在需要覆写的地方去覆写新增方法. 

#### **static**
静态字段: static field.  
实例字段在每个实例中都有自己的一个独立“空间”, 但是静态字段只有一个共享“空间”, 所有实例都会共享该字段
对于静态字段, 无论修改哪个实例的静态字段, 效果都是一样的: 所有实例的静态字段都被修改了, 原因是静态字段并不属于实例

不推荐用 '实例变量.静态字段' 去访问静态字段, 因为在Java程序中, 实例对象并没有静态字段. 
在代码中, 实例对象能访问静态字段只是因为编译器可以根据实例类型自动转换为 '类名.静态字段' 来访问静态对象(会得到一个编译警告).  
`ming.number = 88;`  
`hong.number = 99;`  
推荐用类名来访问静态字段. 可以把静态字段理解为描述class本身的字段(非实例字段).
`Person.number = 99;`  

调用实例方法必须通过一个实例变量, 而调用静态方法则不需要实例变量, 通过类名就可以调用. 
因为静态方法属于class而不属于实例, 因此, 静态方法内部, 无法访问this变量, 也无法访问实例字段, 它只能访问静态字段  
`Person.setNumber(99);`  
静态方法常用于工具类和辅助方法. 

因为interface是一个纯抽象类, 所以它不能定义实例字段. 但是, interface是可以有静态字段的, 并且静态字段必须为final类型. 
实际上, 因为interface的字段只能是public static final类型, 所以我们可以把这些修饰符都去掉
编译器会自动加上public statc final:
```java
public interface Person {
    int MALE = 1;
    int FEMALE = 2;
}
```

#### 包
Java定义了一种名字空间, 称之为包: package. 一个类总是属于某个包, 类名(比如Person)只是一个简写, 真正的完整类名是包名.类名
在Java虚拟机执行的时候, JVM只看完整类名, 因此, 只要包名不同, 类就不同
特别注意: 包没有父子关系. java.util和java.util.zip是不同的包, 两者没有任何继承关系. 
位于同一个包的类, 可以访问包作用域的字段和方法. 不用public、protected、private修饰的字段和方法就是包作用域

在写import的时候, 可以使用*, 表示把这个包下面的所有class都导入进来(但不包括子包的class)
导入mr.jun包的所有class:
import mr.jun.*;

Java编译器最终编译出的.class文件只使用完整类名, 因此, 在代码中, 当编译器遇到一个class名称时: 
如果是完整类名, 就直接根据完整类名查找这个class；
如果是简单类名, 按下面的顺序依次查找: 
1. 查找当前package是否存在这个class；
2. 查找import的包是否包含这个class；
3. 查找java.lang包是否包含这个class

因此, 编写class的时候, 编译器会自动帮我们做两个import动作: 
1. 默认自动import当前package的其他class；
2. 默认自动import java.lang.*. 
注意: 自动导入的是java.lang包, 但类似java.lang.reflect这些包仍需要手动导入

#### **作用域**
由于Java支持嵌套类, 如果一个类内部还定义了嵌套类, 那么, 嵌套类拥有访问private的权限.

protected:   
protected作用于继承关系. 定义为protected的字段和方法可以被子类访问, 以及子类的子类.

package:  
包作用域是指一个类允许访问同一个package的没有public、private修饰的class, 以及没有public、protected、private修饰的字段和方法. 
只要在同一个包, 就可以访问package权限的class、field和method

final:  
final与访问权限不冲突, 它有很多作用
* 用final修饰class可以阻止被继承
* 用final修饰method可以阻止被子类覆写
* 用final修饰field可以阻止被重新赋值
* 用final修饰局部变量可以阻止被重新赋值

一个.java文件只能包含一个public类, 但可以包含多个非public类. 如果有public类, 文件名必须和public类的名字相同.

#### **内部类**
就是内部类Inner Class的实例不能单独存在, 必须依附于一个Outer Class的实例.
因为Inner Class除了有一个this指向它自己, 还隐含地持有一个Outer Class实例, 可以用Outer.this访问这个实例. 
所以, 实例化一个Inner Class不能脱离Outer实例. 

Inner Class和普通Class相比, 除了能引用Outer实例外, 还有一个额外的“特权”, 就是可以修改Outer Class的private字段, 
因为Inner Class的作用域在Outer Class内部, 所以能访问Outer Class的private字段和方法.

还有一种定义Inner Class的方法, 它不需要在Outer Class中明确地定义这个Class, 而是在方法内部, 通过匿名类(Anonymous Class)来定义.
匿名类和Inner Class一样, 可以访问Outer Class的private字段和方法
之所以我们要定义匿名类, 是因为在这里我们通常不关心类名, 比直接定义Inner Class可以少写很多代码.

观察Java编译器编译后的.class文件可以发现, Outer类被编译为Outer.class, 而匿名类被编译为Outer$1.class. 
如果有多个匿名类, Java编译器会将每个匿名类依次命名为Outer$1、Outer$2、Outer$3

用static修饰的内部类和Inner Class有很大的不同, 它不再依附于Outer的实例, 而是一个完全独立的类, 
因此无法引用Outer.this, 但它可以访问Outer的private静态字段和静态方法. 

### 反射
Java的反射是指程序在运行期可以拿到一个对象的所有信息. 反射是为了解决在运行期, 对某个实例一无所知的情况下, 如何调用其方法

### 注解(Annotation)
注解则可以被编译器打包进入class文件, 因此, 注解是一种用作标注的“元数据”
Java的注解可以分为三类: 
1. 第一类是由编译器使用的注解, 例如: 
    * @Override: 让编译器检查该方法是否正确地实现了覆写；
    * @SuppressWarnings: 告诉编译器忽略此处代码产生的警告

    这类注解不会被编译进入.class文件, 它们在编译后就被编译器扔掉了

2. 第二类是由工具处理.class文件使用的注解, 有些工具会在加载class的时候, 对class做动态修改, 实现一些特殊的功能. 这类注解会被编译进入.class文件, 但加载结束后并不会存在于内存中. 这类注解只被一些底层库使用, 一般不必自己处理. 

3. 第三类是在程序运行期能够读取的注解, 它们在加载后一直存在于JVM中, 这也是最常用的注解. 例如, 一个配置了@PostConstruct的方法会在调用构造方法后自动被调用(这是Java代码读取该注解实现的功能, JVM并不会识别该注解)

Java语言使用@interface语法来定义注解(Annotation)


### 单元测试