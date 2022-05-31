# 深度学习







# Python

#### 

### 闭包

- **函数嵌套**
- **内部函数使用外部函数的变量**
- **外部函数的返回值为内部函数**
- 返回的函数并没有立刻执行，而是直到调用了f()才执行
- 返回函数不要引用任何循环变量，或者后续会发生变化的变量

```python
# 如果不需要立刻求和，而是在后面的代码中，根据需要再计算怎么办？可以不返回求和的结果，而是返回求和的函数：
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum
  
  
# 当我们调用lazy_sum()时，返回的并不是求和结果，而是求和函数：
f = lazy_sum(1, 3, 5, 7, 9)  # <function lazy_sum.<locals>.sum at 0x101c6ed90>


# 调用函数f时，才真正计算求和的结果：
f()  # 25


- 我们在函数lazy_sum中又定义了函数sum，并且，内部函数sum可以引用外部函数lazy_sum的参数和局部变量，当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中，这种称为“闭包（Closure）”;

- 返回的函数并没有立刻执行，而是直到调用了f()才执行;

- 当我们调用lazy_sum()时，每次调用都会返回一个新的函数，即使传入相同的参数：
f1 = lazy_sum(1, 3, 5, 7, 9)
f2 = lazy_sum(1, 3, 5, 7, 9)
f1 == f2  # False,f1()和f2()的调用结果互不影响

- 返回函数不要引用任何循环变量，或者后续会发生变化的变量;
```



### 装饰器



- 装饰器,被装饰函数都不带参数
- 带参数的被装饰的函数
- 带参数的装饰器,装饰函数

```python
# 不带参数的装饰器
import time
def showtime(func):
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        print('spend is {}'.format(end_time - start_time))

    return wrapper

@showtime  #foo = showtime(foo)
def foo():
    print('foo..')
    time.sleep(3)

@showtime #doo = showtime(doo)
def doo():
    print('doo..')
    time.sleep(2)

foo()
doo()
```

```python
# 带参数的被装饰的函数
import time
def showtime(func):
    def wrapper(a, b):
        start_time = time.time()
        func(a,b)
        end_time = time.time()
        print('spend is {}'.format(end_time - start_time))

    return wrapper

@showtime #add = showtime(add)
def add(a, b):
    print(a+b)
    time.sleep(1)

@showtime #sub = showtime(sub)
def sub(a,b):
    print(a-b)
    time.sleep(1)

add(5,4)
sub(3,2)
```

```python
# 带参数的装饰器,装饰函数
import time
def time_logger(flag = 0):
    def showtime(func):
        def wrapper(a, b):
            start_time = time.time()
            func(a,b)
            end_time = time.time()
            print('spend is {}'.format(end_time - start_time))

            if flag:
                print('将此操作保留至日志')

        return wrapper

    return showtime

@time_logger(2)  #得到闭包函数showtime,add = showtime(add)
def add(a, b):
    print(a+b)
    time.sleep(1)

add(3,4)
```

