快速开始
========

安装依赖
--------

框架最小依赖：

.. code-block:: bash

   pip install PyYAML>=6.0 "ray[default]"

如果需要运行内置的 SUMO 示例（``sumo_ryl``），还需要额外安装：

.. code-block:: bash

   pip install torch
   # 另需在本机安装 SUMO，并确保 traci / sumolib 可用

运行框架
--------

基础命令：

.. code-block:: bash

   python run.py --config configs/default.yaml

运行 SUMO 示例：

.. code-block:: bash

   python run.py --config configs/sumo_ryl.yaml

``run.py`` 入口逻辑非常简单：

.. code-block:: python

   # run.py
   from src.config.loader import load_config
   from src.pipeline.runner import run_masdiff

   cfg = load_config(args.config)
   run_masdiff(cfg)

输出说明
--------

运行后会在 ``outputs/`` 目录下生成：

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - 文件
     - 内容
   * - ``best_rho_history.csv``
     - 每轮迭代种群最优 ρ，字段为 ``iteration_k, best_rho``
   * - ``*_timings.csv``
     - 各阶段耗时统计，用于性能分析
   * - ``q_*.pt``
     - Q 的缓存文件（如果模块实现了缓存）

.. note::

   CSV 文件采用追加写入。反复运行同一配置时不会自动清空，需手动删除旧文件。
