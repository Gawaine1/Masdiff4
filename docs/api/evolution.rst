EliteSelector 接口
==================

定义位置：``src/evolution/base.py``

职责
----

``EliteSelector`` 根据种群个体的适应度 ``ρ`` 选择精英子集，用于后续的截断扩散变异。

抽象方法
--------

select_elites
~~~~~~~~~~~~~

.. code-block:: python

   def select_elites(
       self,
       population: Population,
       *,
       elite_count: int
   ) -> Population:
       ...

- **输入** ``population``：当前种群；
- **输入** ``elite_count``：需要选择的精英数量（来自配置文件 ``elite.elite_count``）；
- **输出**：精英个体列表。

选择策略完全由用户自定义，可以是确定性的 top-k，也可以是带探索性的随机采样。

YAML 中声明
-----------

.. code-block:: yaml

   elite_selector:
     class_path: "your_pkg.evolution:YourEliteSelector"
     kwargs: {}

内置实现
--------

- ``src/evolution/temperature_selection.py`` → ``TemperatureEliteSelector``

  使用 softmax 温度采样：

  .. code-block:: text

     probs = softmax(temperature * rho_scores)
     使用 torch.multinomial 采样 elite_count 个精英

  特点：精英不是绝对 top-k，而是带有一定探索概率，避免早熟收敛。
