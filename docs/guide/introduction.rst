项目简介
========

MASDiff 是什么
--------------

MASDiff 是一个围绕"多智能体仿真 + DQN 策略训练 + 条件扩散奖励生成 + 进化迭代优化"组织起来的实验框架。

这个项目的核心定位不是提供某一个固定场景下的完整算法产品，而是提供：

- 一套**固定主流程**，对应 ``主流程.txt`` 中定义的 1～5.6 步；
- 一组可替换的**抽象接口**（Environment、DqnModule、DiffusionModel 等）；
- 一种基于 **YAML + Python 动态导入** 的模块拼装方式；
- 一套已经落地的 **SUMO 路网仿真示例实现（sumo_ryl）**。

框架负责调度流程，具体算法由用户实现并在 YAML 中声明。

解决的问题
----------

MASDiff 关注以下问题：

1. 在多智能体环境中先随机生成一批策略；
2. 通过环境仿真收集状态信息 ``Tau`` 和经验库；
3. 用扩散模型根据 ``Tau`` 生成奖励 ``R``；
4. 用 ``R`` 反向构建 DQN 训练数据并训练策略；
5. 用训练后的策略再次仿真，计算每个个体相对于目标 ``Q`` 的指标 ``ρ``；
6. 将 ``(Tau, R, ρ)`` 作为种群个体，持续做扩散训练、精英选择、截断扩散变异和保优迭代。

整体架构
--------

项目分为四层：

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - 层次
     - 关键文件
     - 职责
   * - 入口层
     - ``run.py``
     - 读取配置，调用主流程
   * - 配置层
     - ``src/config/schema.py``, ``src/config/loader.py``
     - YAML 解析、类型化、校验
   * - 流程调度层
     - ``src/pipeline/runner.py``, ``src/pipeline/steps.py``
     - 算法主流程分解与执行
   * - 可插拔模块层
     - ``src/environments/``, ``src/dqn/``, ``src/diffusion/`` 等
     - 用户自定义或内置示例实现

目录结构
--------

.. code-block:: text

   MASDiff/
   ├─ run.py                         # 程序入口
   ├─ README.md
   ├─ requirements.txt
   ├─ configs/                       # YAML 配置文件
   ├─ maps/                          # SUMO 路网/路由/仿真配置
   ├─ outputs/                       # 输出目录
   └─ src/
      ├─ config/                     # 配置模型与加载器
      ├─ pipeline/                   # 主流程、步骤拆分、数据类型
      ├─ environments/               # 环境接口与 SUMO 实现
      ├─ dqn/                        # DQN 接口与示例实现
      ├─ diffusion/                  # 扩散模型接口与示例实现
      ├─ evolution/                  # 精英选择接口与示例实现
      ├─ q/                          # Q 的读取/生成接口与示例实现
      ├─ metrics/                    # ρ 指标接口与示例实现
      ├─ parallel/                   # 串行/Ray 并行执行器
      └─ utils/                      # 动态导入工具
