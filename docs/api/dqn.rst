DqnModule 接口
==============

定义位置：``src/dqn/base.py``

职责
----

``DqnModule`` 是 DQN 策略模块的抽象基类，负责：

- 初始化随机策略（N 个智能体各自一个 DQN）；
- 将经验库和扩散模型生成的奖励合并，构建训练数据；
- 为每个智能体独立训练一个 DQN 策略。

抽象方法
--------

init_random_policies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def init_random_policies(self, num_agents: int) -> list[Any]:
       ...

初始化随机策略：

- **输入** ``num_agents``：智能体数量 ``N``；
- **输出** ``policies``：长度为 ``N`` 的随机策略列表。

build_training_data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def build_training_data(
       self,
       experience_buffers: list[list[Any]],
       rewards: Any
   ) -> Any:
       ...

用奖励填充经验，构建 DQN 训练数据：

- **输入** ``experience_buffers``：各智能体的经验序列（``r`` 字段留空）；
- **输入** ``rewards``：扩散模型生成的奖励矩阵；
- **输出** ``training_data``：可直接用于训练的数据结构，格式由用户定义。

train_per_agent
~~~~~~~~~~~~~~~

.. code-block:: python

   def train_per_agent(self, training_data: Any) -> list[Any]:
       ...

为每个智能体训练 DQN：

- **输入** ``training_data``：``build_training_data`` 的输出；
- **输出** ``policies``：训练完成的策略列表，长度为 ``N``。

YAML 中声明
-----------

.. code-block:: yaml

   dqn_module:
     class_path: "your_pkg.dqn:YourDqnModule"
     kwargs:
       hidden_dim: 128
       lr: 0.001

内置实现
--------

- ``src/dqn/sumo_ryl_dqn.py`` → ``SumoRylDqnModule``

  每个智能体对应独立的 DQN 模型，使用 ``(s, a, r)`` 做单步回归训练。
  奖励填充策略支持按目标路段索引（``to``）或起始路段索引（``from``）两种方式。
  详见 :doc:`../guide/sumo_ryl`。
