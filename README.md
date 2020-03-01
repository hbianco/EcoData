# EcoData
EcoData is a library to perform usual transformations on financial and economic times series such as YoY or Lead/Lag.
Users can also display results in clear charts.

## Examples

1. Display price series with moving averages, RSI and trading signals.
```python
import EcoData as ed

ed.graph(data, ma_vs, ma_s, ma_l, chart_size=(15,10), legend=False, 
         subgraph=rsi_l, trading_signal=trig, view_grid=True)
```
![alt text](/static/eg2.png)
  

2. Without moving averages.
```python
ed.graph(data, chart_size=(15,10), legend=False, subgraph=r_l,
         trading_signal=t, view_grid=True)
```
![alt text](/static/eg3.png)
  

3. Add P&L line on the chart and save the charts as "strategy_pnl.png".
```python
ed.graph(sp, pnl, multiple_series=True, chart_size=(15,10), legend=False, 
         subgraph=sp_rsi, trading_signal=trig, view_grid=True, save_fig="strategy_pnl.png")
```
![alt text](/static/eg1.png)
  

4. Create a candle chart with RSI indicator and save it as "candle_chart.png".

```python
ed.graph(eurusd[-101:], chart_size=(15,10), candle=True, legend=False, weekdays=False, candle_width=0.5,
  subgraph=e_rsi[-101:], view_grid=True, save_fig="candle_chart.png")
```
![alt text](/static/eg4.png)

