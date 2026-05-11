算法主流程
==========

MASDiff 的完整算法主流程定义在 ``主流程.txt`` 中，由 ``src/pipeline/runner.py`` 和 ``src/pipeline/steps.py`` 实现。

流程总览
--------

.. code-block:: text

   0. 初始化：动态加载各模块
   1. 读取或创建 Q
   2. 随机初始化扩散模型
   3~4. 构建初始种群（M 个个体）
       └─ 4.1 随机策略仿真，收集经验库和 Tau
       └─ 4.2 扩散模型根据 Tau 生成奖励 R
       └─ 4.3 经验库 + R → DQN 训练数据
       └─ 4.4 训练每个智能体的 DQN
       └─ 4.5 再次仿真，计算 ρ
       └─ 个体 = (Tau, R, ρ)
   5. 进化迭代 K 轮
       └─ 5.1 用种群训练扩散模型
       └─ 5.2 选择精英种群
       └─ 5.3 遍历精英，生成变异个体
           └─ 5.3.1~5.3.8 截断扩散变异 + 两轮 DQN 训练
       └─ 5.4 变异个体加入种群
       └─ 5.5 保留 top-M 个体
       └─ 5.6 记录最优 ρ 到 CSV

----

步骤 0：初始化模块
------------------

根据 YAML 中的 ``class_path`` 动态实例化以下模块，并做类型检查：

- ``q_provider`` → 继承 ``QProvider``
- ``environment`` → 继承 ``Environment``
- ``dqn_module`` → 继承 ``DqnModule``
- ``diffusion_model`` → 继承 ``DiffusionModel``
- ``elite_selector`` → 继承 ``EliteSelector``
- ``metric`` → 继承 ``Metric``
- ``executor`` → 继承 ``ParallelExecutor``

步骤 1：读取或创建 Q
--------------------

.. code-block:: python

   q = step_1_load_or_create_q(q_provider, environment)
   # 调用：q_provider.load_or_create_q(environment)

``Q`` 是评价基准（目标、参考轨迹或标签），具体结构由用户自定义。
例如在 SUMO 场景中，``Q`` 是纯 A* 基线下的排队长度矩阵。

步骤 2：随机初始化扩散模型
--------------------------

.. code-block:: python

   diffusion_model = step_2_init_diffusion_model(diffusion_model)
   # 调用：diffusion_model.init_random()

步骤 3～4：构建初始种群
------------------------

设种群规模为 ``M``，每个个体包含 ``N`` 个智能体。
对每个个体 ``i = 1..M`` 依次执行：

**4.1 仿真收集**

.. code-block:: python

   experience_buffers, tau = step_4_1_simulate_collect(environment, init_policies)
   # 调用：environment.simulate_collect(policies)

- ``experience_buffers``：每个智能体的经验序列，单条经验格式为 ``[s, a, r]``，``r`` 暂时留空；
- ``tau``：描述环境状态的条件向量（如路网特征），作为扩散模型的生成条件。

**4.2 生成奖励 R**

.. code-block:: python

   rewards = step_4_2_generate_reward(diffusion_model, tau)
   # 调用：diffusion_model.generate_reward(tau)

扩散模型以 ``tau`` 为条件生成奖励矩阵 ``R``。

**4.3 构建 DQN 训练数据**

.. code-block:: python

   training_data = step_4_3_build_dqn_training_data(dqn_module, experience_buffers, rewards)
   # 调用：dqn_module.build_training_data(experience_buffers, rewards)

用奖励 ``R`` 填充经验中的 ``r`` 字段，形成完整训练样本。

**4.4 训练 DQN**

.. code-block:: python

   trained_policies = step_4_4_train_dqn_per_agent(dqn_module, training_data)
   # 调用：dqn_module.train_per_agent(training_data)

为每个智能体独立训练一个 DQN 策略。

**4.5 再次仿真，计算 ρ**

.. code-block:: python

   simulation_data, rho = step_4_5_simulate_and_compute_rho(
       environment, trained_policies, q=q, metric=metric
   )
   # 调用：environment.simulate_evaluate(policies)
   #       metric.compute_rho(q, simulation_data)

``ρ`` 是适应度指标，越大越好。

**汇总个体**

.. code-block:: python

   ind = step_4_build_individual(
       tau=tau, rewards=rewards, rho=rho,
       experience_buffers=experience_buffers, policies=trained_policies
   )

初始种群构建支持通过 ``ParallelExecutor`` 并行化：

.. code-block:: python

   population = executor.map(build_initial_individual, list(range(1, M + 1)))

----

步骤 5：进化迭代（K 轮）
-------------------------

**5.1 训练扩散模型**

.. code-block:: python

   diffusion_model = step_5_1_train_diffusion_with_population(diffusion_model, population)
   # 调用：diffusion_model.train_on_population(population)

用当前种群中全部 ``(tau, rewards)`` 对训练扩散模型。

**5.2 选择精英种群**

.. code-block:: python

   elites = step_5_2_select_elite_population(elite_selector, population, elite_count=cfg.elite.elite_count)
   # 调用：elite_selector.select_elites(population, elite_count=...)

精英选择策略完全由用户实现（如 top-k、温度采样等）。

**5.3 精英变异（截断扩散）**

对每个精英个体依次执行以下 8 个子步骤：

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - 子步骤
     - 说明
   * - 5.3.1
     - 以精英的 ``tau`` 为条件，对精英原有奖励 ``R`` 做截断扩散变异（先加噪少量步，再去噪少量步），得到变异奖励
   * - 5.3.2
     - 将精英保存的经验库与变异奖励合并，构建 DQN 训练数据
   * - 5.3.3
     - 训练每个智能体的 DQN（第一轮训练）
   * - 5.3.4
     - 用第一轮训练策略仿真，重新收集经验库和新的 ``tau``
   * - 5.3.5
     - 以新 ``tau`` 为条件，扩散模型生成新奖励 ``R``
   * - 5.3.6
     - 将新经验库与新奖励合并，构建 DQN 训练数据
   * - 5.3.7
     - 再次训练每个智能体的 DQN（第二轮训练，得到最终策略）
   * - 5.3.8
     - 用最终策略仿真并计算 ``ρ``，形成变异个体

截断扩散是实现"**围绕优秀个体做局部探索**"的关键，加噪/去噪步数通过配置控制变异幅度。

精英变异同样支持并行化：

.. code-block:: python

   mutants = executor.map(mutate_one, elites)

**5.4 合并种群**

.. code-block:: python

   merged_population = step_5_4_add_mutants_to_population(population, mutants)

**5.5 保留 top-M**

.. code-block:: python

   population = step_5_5_keep_top_m(merged_population, M=M)

按 ``ρ`` 从大到小排序，只保留前 ``M`` 个个体。

**5.6 记录最优 ρ**

.. code-block:: python

   population, best_rho_k = step_5_6_record_best_rho(
       population, iteration_k=k, csv_path=cfg.logging.best_rho_csv_path
   )

将本轮最优 ``ρ`` 追加写入 CSV 文件。

----

steps.py 的设计意图
--------------------

``src/pipeline/steps.py`` 将主流程拆分为一系列标准步骤函数，例如：

- ``step_1_load_or_create_q``
- ``step_4_1_simulate_collect``
- ``step_5_3_1_truncated_diffusion_mutate_reward``
- ``step_5_6_record_best_rho``

这样做的好处：

- 主流程（``runner.py``）可读性高，每行代码对应算法的一步；
- 方便替换某一步的具体实现；
- 方便编写单步单元测试；
- 步骤函数可在不同流程中复用（如 5.3.2 复用了 4.3 的实现）。
