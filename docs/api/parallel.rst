ParallelExecutor 接口
=====================

定义位置：``src/parallel/base.py``

职责
----

``ParallelExecutor`` 为以下两处计算密集段提供统一的并行（或串行）接口：

1. 构建初始种群（步骤 3～4，``M`` 个个体）；
2. 精英变异（步骤 5.3，``elite_count`` 个精英）。

通过接口隔离，框架无需感知底层是串行、多进程还是 Ray 分布式。

抽象方法
--------

map
~~~

.. code-block:: python

   def map(self, fn: Callable, items: list) -> list:
       ...

对 ``items`` 中的每个元素调用 ``fn``，返回结果列表：

- **输入** ``fn``：待执行的函数（如 ``build_initial_individual`` 或 ``mutate_one``）；
- **输入** ``items``：输入列表；
- **输出**：与 ``items`` 等长的结果列表。

close
~~~~~

.. code-block:: python

   def close(self) -> None:
       ...

释放执行器资源（如关闭 Ray 连接）。框架在 ``finally`` 块中调用此方法确保资源释放。

YAML 中声明
-----------

.. code-block:: yaml

   parallel_executor:
     class_path: "src.parallel.serial:SerialExecutor"
     kwargs: {}

内置实现
--------

SerialExecutor
~~~~~~~~~~~~~~

定义位置：``src/parallel/serial.py``

- 默认执行器，不做任何并行处理；
- 保留统一的 ``map`` 接口；
- 支持简单进度打印；
- 适合：开发调试、小规模实验、未部署 Ray 的环境。

.. code-block:: yaml

   parallel_executor:
     class_path: "src.parallel.serial:SerialExecutor"
     kwargs: {}

RayExecutor
~~~~~~~~~~~

定义位置：``src/parallel/ray_executor.py``

- 基于 `Ray <https://www.ray.io/>`_ 的分布式并行执行器；
- 可为不同函数名配置不同资源（CPU/GPU 数量）；
- 适合大规模仿真与训练。

.. code-block:: yaml

   parallel_executor:
     class_path: "src.parallel.ray_executor:RayExecutor"
     kwargs:
       ray_init_kwargs:
         num_cpus: 25
         num_gpus: 3
       task_options_by_name:
         build_initial_individual:
           num_cpus: 2.5
           num_gpus: 0.3
         mutate_one:
           num_cpus: 2.5
           num_gpus: 0.3

Ray 大对象传输优化
~~~~~~~~~~~~~~~~~~

为避免 Ray 序列化闭包中的大对象（模型权重等），框架做了以下处理：

- 将并行任务封装为顶层函数（``src/parallel/ray_tasks.py``），避免闭包序列化膨胀；
- 使用 ``ray.put(q)``、``ray.put(diffusion_state)`` 将大对象放入 Ray object store，由 worker 端引用而非复制；
- ``Individual`` 默认不保存训练后的大策略对象（``policies = []``）。
