"""第14讲 库存周转与补货建议
业务问题：销售分析如何变成补货量？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

# %% 验证库存流水，而不是只相信最后一列
t=load_tables();inv=t['inventory']
assert (inv.opening_qty+inv.inbound_qty+inv.restock_qty-inv.shipped_qty==inv.closing_qty).all()
recent=inv.loc[inv.inventory_date.ge(AS_OF-pd.Timedelta(days=30))]
demand=recent.groupby('product_id').agg(mean_daily=('shipped_qty','mean'),sd_daily=('shipped_qty','std'))
latest=inv.sort_values('inventory_date').groupby('product_id').tail(1)[['product_id','closing_qty']]
plan=latest.merge(demand,on='product_id',validate='one_to_one').merge(t['products'],on='product_id',validate='one_to_one')
lead_days=5;review_days=7;z=1.65  # 教学假设，未按真实缺货成本优化
plan['safety_stock']=np.ceil(z*plan.sd_daily*np.sqrt(lead_days)).astype(int)
plan['reorder_point']=np.ceil(plan.mean_daily*lead_days+plan.safety_stock).astype(int)
plan['target_stock']=np.ceil(plan.mean_daily*(lead_days+review_days)+plan.safety_stock).astype(int)
plan['suggested_order_qty']=(plan.target_stock-plan.closing_qty).clip(lower=0)
plan['stock_cover_days']=safe_divide(plan.closing_qty,plan.mean_daily)
plan['stock_value_yuan']=plan.closing_qty*plan.unit_cost_cents/100
show(plan[['product_id','closing_qty','mean_daily','safety_stock','reorder_point','suggested_order_qty']])
save_table(plan,14,'replenishment_plan')
lines,_=build_facts(t)
average=inv.assign(avg_units=(inv.opening_qty+inv.closing_qty)/2).groupby('product_id').avg_units.mean()
annual_cogs=lines.groupby('product_id').cogs_cents.sum()
turnover=t['products'][['product_id','unit_cost_cents']].copy()
turnover=turnover.merge(average.rename('average_inventory_units'),on='product_id',validate='one_to_one')
turnover=turnover.merge(annual_cogs.rename('annual_cogs_cents'),on='product_id',validate='one_to_one')
turnover['average_inventory_cost_cents']=turnover.average_inventory_units*turnover.unit_cost_cents
turnover['annual_turnover']=safe_divide(turnover.annual_cogs_cents,turnover.average_inventory_cost_cents)
turnover['inventory_days']=safe_divide(pd.Series(365,index=turnover.index),turnover.annual_turnover)
save_table(turnover,14,'annual_inventory_turnover');show(turnover)
print('平均库存成本用每天期初和期末均值再取全年平均近似；周转分子是成本，不是销售收入。')
fig,ax=plt.subplots(figsize=(9,4));ax.scatter(plan.mean_daily,plan.stock_cover_days,color=BLUE,s=30)
for row in plan.itertuples():ax.annotate(row.product_id,(row.mean_daily,row.stock_cover_days),fontsize=6,xytext=(2,2),textcoords='offset points')
ax.set(title='Stock cover and recent gross demand',xlabel='Mean daily shipped units',ylabel='Stock cover (days)');save_figure(fig,14,'stock_cover')
print('需求稳定、提前期固定、无在途订单约束是本例简化；补货建议不能直接下采购单。')
