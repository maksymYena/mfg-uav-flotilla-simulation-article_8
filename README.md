# Modeling Collective Behavior of UAV Flotillas Using Mean Field Games (Article 8)

This repository contains the simulation code, visualization tools, and comparative metrics for analyzing the collective behavior of unmanned aerial vehicle (UAV) flotillas using **Mean Field Game (MFG)** theory versus a **rule-based** baseline model. This work supports Article 8 in the author's research on intelligent traffic management for UAVs.

---

## 🧠 Key Concepts

- **Mean Field Games (MFG)**: A mathematical framework to model large populations of agents (UAVs) making decisions based on both individual goals and collective dynamics.
- **Rule-Based Model**: A deterministic heuristic approach where each agent moves directly towards the goal, ignoring density and potential conflicts.
- **Obstacle Avoidance**: Both models consider a circular forbidden zone to test their ability to adapt to environmental constraints.
- **Metric Comparison**: The performance of both models is evaluated using metrics like average distance to target, congestion, obstacle penetration, mass conservation, and adaptation to the environment.

---

## 🧪 What’s Inside?

- `simulate_mfg()`: Implements the HJB-FP system for MFG.
- `simulate_rule_based()`: Implements a heuristic baseline model.
- `plot_density_snapshot(...)`: Visualizes agent density and potential fields.
- `average_distance`, `congestion_metric`, `mass_in_obstacle`: Key evaluation functions.
- Final performance results are printed as pandas `DataFrame` tables and heatmap plots.

---

## 📊 Example Metrics

| Метрика                                       | Модель масових ігор (MFG) | Rule-Based модель |
|----------------------------------------------|----------------------------|-------------------|
| Середня відстань до цілі                     | 6.41                       | 10.25             |
| Максимальна локальна густина (конфлікти)     | 0.0274                     | 0.0546            |
| Маса агентів у перешкодах                    | 0.0000                     | 0.0012            |
| Час виконання симуляції (сек)                | 2.893                      | 1.605             |
| Адаптація напрямку до цілі (cos θ)           | 0.9912                     | 0.8451            |
| Втрати маси при симуляції                    | 0.0000                     | 0.0123            |

> ✅ MFG demonstrates better congestion handling and environmental adaptation.  
> ⚠️ Rule-based agents often collide with obstacles and show less efficient density control.

---

## 🖼️ Visual Examples

- Heatmaps of agent distribution
- Quiver plots showing movement direction
- Delta-density plot comparing the two models directly

---

## 🧩 Requirements

- Python 3.8+
- Required packages:
  - `numpy`
  - `matplotlib`
  - `pandas`

Install dependencies:
```bash
pip install -r requirements.txt
