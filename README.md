# EcoData
EcoData is a library to perform usual transformations on financial and economic times series such as YoY or Lead/Lag.
Users can also display results in clear charts.

## Examples

```python
import EcoData as ed

ed.graph(eurusd[-101:], chart_size=(15,10), candle=True, legend=False, weekdays=False, candle_width=0.5,
  subgraph=e_rsi[-101:], view_grid=True, save_fig="t.png")
```
![alt text](/static/eg4.png)

