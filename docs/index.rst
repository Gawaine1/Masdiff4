.. MASDiff documentation master file

MASDiff 文档
============

.. image:: https://img.shields.io/badge/Python-3.9+-blue
.. image:: https://img.shields.io/badge/License-MIT-green

**MASDiff** 是一个以"多智能体仿真 + DQN 策略训练 + 条件扩散奖励生成 + 进化迭代优化"为核心的可插拔实验框架。

框架负责把算法主流程串联起来，而环境、Q 的构造、DQN 训练、扩散模型、精英选择、评价指标、并行执行器都可以由用户自行实现并接入。

.. tip::

   MASDiff 本质上是一个"**扩散模型辅助奖励生成的种群式策略搜索框架**"。
   它的核心定位是：固定主流程 + 可替换抽象接口 + YAML 模块拼装。

----

.. toctree::
   :maxdepth: 2
   :caption: 入门指南

   guide/introduction
   guide/quickstart
   guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: 核心概念

   guide/algorithm
   guide/data_structures

.. toctree::
   :maxdepth: 2
   :caption: 模块接口

   api/overview
   api/environment
   api/dqn
   api/diffusion
   api/evolution
   api/metrics
   api/parallel

.. toctree::
   :maxdepth: 2
   :caption: 内置示例

   guide/sumo_ryl

.. toctree::
   :maxdepth: 2
   :caption: 开发扩展

   guide/extending

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
