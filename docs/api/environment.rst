Environment 接口
================

定义位置：``src/environments/base.py``

职责
----

``Environment`` 是仿真环境的抽象基类，负责：

- 运行多智能体仿真；
- 收集经验库（experience buffers）；
- 产出 ``Tau``（条件状态向量）；
- 产出用于评价的仿真结果（simulation data）。

抽象方法
--------

simulate_collect
~~~~~~~~~~~~~~~~

.. code-block:: python

   def simulate_collect(self, policies: list[Any]) -> tuple[list[list[Any]], Any]:
       ...

用给定策略运行仿真，收集数据：

- **输入** ``policies``：长度为 ``N`` 的策略列表，每个元素对应一个智能体；
- **输出** ``experience_buffers``：每个智能体的经验序列，单条经验格式为 ``[s, a, r]``，``r`` 留空；
- **输出** ``tau``：描述仿真状态的条件向量，作为扩散模型生成奖励的输入。

simulate_evaluate
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def simulate_evaluate(self, policies: list[Any]) -> Any:
       ...

用给定策略运行仿真，产出评价数据：

- **输入** ``policies``：同上；
- **输出** ``simulation_data``：评价用的仿真输出，格式由用户定义，需与 ``Metric.compute_rho()`` 的输入约定保持一致。

YAML 中声明
-----------

.. code-block:: yaml

   environment:
     class_path: "your_pkg.envs:YourEnvironment"
     kwargs:
       some_param: value

内置实现
--------

框架提供了基于 SUMO 路网仿真的示例实现：

- ``src/environments/sumo_ryl.py`` → ``SumoRylEnvironment``

详见 :doc:`../guide/sumo_ryl`。
