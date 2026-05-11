配置系统
========

MASDiff 使用 YAML 文件组织完整的实验配置，一份 YAML 描述一整套实验。

配置文件格式
------------

以下是典型的配置结构示意：

.. code-block:: yaml

   seed: 42

   algorithm:
     M: 10          # 种群规模
     N: 5           # 每个个体中的智能体数量
     K: 3           # 进化迭代次数

   elite:
     elite_count: 3 # 每轮精英选择数量

   truncated_diffusion:
     add_noise_steps: 10   # 截断扩散加噪步数
     denoise_steps: 10     # 截断扩散去噪步数

   logging:
     best_rho_csv_path: "outputs/best_rho_history.csv"

   # 各模块通过 class_path 指向用户实现
   environment:
     class_path: "your_pkg.envs:YourEnvironment"
     kwargs: {}

   dqn_module:
     class_path: "your_pkg.dqn:YourDqnModule"
     kwargs: {}

   diffusion_model:
     class_path: "your_pkg.diffusion:YourDiffusionModel"
     kwargs: {}

   elite_selector:
     class_path: "your_pkg.evolution:YourEliteSelector"
     kwargs: {}

   q_provider:
     class_path: "your_pkg.q:YourQProvider"
     kwargs: {}

   metric:
     class_path: "your_pkg.metrics:YourMetric"
     kwargs: {}

   parallel_executor:
     class_path: "src.parallel.serial:SerialExecutor"
     kwargs: {}

关键参数说明
------------

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - 参数
     - 类型
     - 说明
   * - ``algorithm.M``
     - int
     - 种群规模，即同时维护多少个 ``(Tau, R, ρ)`` 个体
   * - ``algorithm.N``
     - int
     - 每个个体包含的智能体数量
   * - ``algorithm.K``
     - int
     - 进化迭代次数
   * - ``elite.elite_count``
     - int
     - 每轮从种群中选出的精英个体数量
   * - ``truncated_diffusion.add_noise_steps``
     - int
     - 截断扩散中对精英奖励加噪的步数，越大变异幅度越大
   * - ``truncated_diffusion.denoise_steps``
     - int
     - 截断扩散中去噪的步数
   * - ``logging.best_rho_csv_path``
     - str
     - 最优 ρ 历史记录的 CSV 输出路径

ModuleSpec 机制
---------------

每个模块在 YAML 中均使用 ``class_path`` + ``kwargs`` 的方式声明，框架启动时通过 ``instantiate(spec)`` 动态导入并实例化。

.. code-block:: python

   # src/utils/import_utils.py
   @dataclass(frozen=True)
   class ModuleSpec:
       class_path: str    # 支持 "pkg.mod:ClassName" 或 "pkg.mod.ClassName"
       kwargs: dict

``class_path`` 支持两种格式：

- ``pkg.mod:ClassName``（推荐，冒号分隔）
- ``pkg.mod.ClassName``（点号分隔）

这使得只要实现了约定接口，就可以直接在 YAML 中替换任意模块，无需修改框架代码。

内置配置文件
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 文件
     - 说明
   * - ``configs/default.yaml``
     - 框架演示配置，使用占位类名，展示各字段用法
   * - ``configs/sumo_ryl.yaml``
     - SUMO 路网示例配置（M=100, N=1557, K=1），适合真实实验
   * - ``configs/sumo_ryl_4000.yaml``
     - 大规模车辆版本（N=4000），用于高负载测试

.. warning::

   ``sumo_ryl.yaml`` 中 M=100, N=1557，运算量极大，需要充足的 CPU/GPU/内存资源和 Ray 集群支持。
