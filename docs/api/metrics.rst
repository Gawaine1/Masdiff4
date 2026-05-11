Metric 接口
===========

定义位置：``src/metrics/base.py``

职责
----

``Metric`` 计算每个种群个体的适应度指标 ``ρ``，用于种群排序和精英选择。

抽象方法
--------

compute_rho
~~~~~~~~~~~

.. code-block:: python

   def compute_rho(self, q: Any, simulation_data: Any) -> float:
       ...

- **输入** ``q``：评价基准（由 ``QProvider`` 提供）；
- **输入** ``simulation_data``：仿真结果（由 ``Environment.simulate_evaluate()`` 返回）；
- **输出** ``ρ``：适应度值，越大越好。

.. note::

   ``q`` 和 ``simulation_data`` 的数据格式需要 ``QProvider``、``Environment``、``Metric``
   三者之间自行约定保持一致，框架不做格式校验。

YAML 中声明
-----------

.. code-block:: yaml

   metric:
     class_path: "your_pkg.metrics:YourMetric"
     kwargs: {}

内置实现
--------

- ``src/metrics/sumo_ryl_metric.py`` → ``SumoRylMetric``

  计算 ``Q``（A* 基线排队数据）与 ``simulation_data``（DQN 策略下排队数据）的误差并转换为适应度：

  .. list-table::
     :widths: 20 80
     :header-rows: 1

     * - 变换模式
       - 公式
     * - ``inv1p``（默认）
       - ``ρ = 1 / (1 + MSE)``
     * - ``inv``
       - ``ρ = 1 / MSE``
     * - ``neg``
       - ``ρ = -MSE``
     * - ``exp``
       - ``ρ = exp(-MSE)``
