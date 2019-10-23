---
title: Django-APP1：PuLP求解线性规划
date: 2018-10-24 23:23:16
tags: Django
---

&emsp;&emsp;本文讲述Django-APP1案例的搭建，这个故事背景是随机生成肯德基点单，需要输入总预算，可以选择主食，也可以不选，可以选择是否打包带走，选的话多一个固定的包装费。背后是一个简单的线性规划问题，使用PuLP进行的求解。
&emsp;&emsp;框架架构改编自一个真实的项目，各部分功能都做了简化只保留原理，随便找到了一些KFC的价格就拿来用了。

<!--more-->

&emsp;&emsp;今天中午准备吃KFC，点点啥呢好纠结啊，这样吧不超过50块随机生成些方案我们来挑吧。这是一个简单的线性规划问题，用模型描述如下：
{% asset_img CodeCogsEqn.gif 问题描述 %}
我们以每一种备选的菜品的购买数量为一个变量，目标是在不超过预算的情况下尽可能的总价值接近它，为了增加复杂度，也为了使营养均衡我加个一些条件：
- 每一种菜品最多买两个；
- 主食是直接指定名称或根据选择的价格来筛选选定，当然也可以不选；
- 每次计算只取50%的备选项参与，所以能产生随机的效果；
- 可以勾选打包，打包多1.5包装费。
首先我们需要准备好一张菜单价格基表，这里我是在网上随便找到整理的一些价格数据，直接导入到自带的sqlite数据库中：
{% asset_link ITEM_CATEGORY.xlsx %}
{% codeblock lang:python %}
# 这一部分详细记录在项目的hisrand_gen.py中
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine(r'sqlite:///db.sqlite3', echo=True)
df = pd.read_excel(r'APP1_LP/appfiles/ITEM_CATEGORY.xlsx')
df.to_sql('APP1_LP_ITEM_CATEGORY',engine,if_exists='replace',index=False)
{% endcodeblock %}
主要的业务处理是views.py中的推荐函数:
{% codeblock lang:python %}
def RandRecommand(TTPay):
    # Main recommand function
    RAND_RECOMMAND = pd.DataFrame(columns=['ITEM_CATEGORY','ITEM','ITEM_DESCRIPTION','Qty','PRICE'])
    RAND_RECOMMAND_INDEX = 0
    # Using rand 50% of the choices everytime
    ITEM_CATEGORY_T = ITEM_CATEGORY[ITEM_CATEGORY['ITEM_CATEGORY']!='Staple'].sample(frac=0.5)
    model = pulp.LpProblem("Rand Food", pulp.LpMinimize)
    var = {}
    TTPRICE = ''
    TTNums = ''
    for index in ITEM_CATEGORY_T.index:
        var[index] = pulp.LpVariable(str(index), lowBound=0, cat='Integer')
        TTPRICE += ITEM_CATEGORY_T.loc[index,'PRICE'] * var[index]
        TTNums += var[index]
        model += var[index] <=2
    model += TTPay - TTPRICE    
    model += TTPRICE <= TTPay
    model.solve()
    status = pulp.LpStatus[model.status]

    for key in var:
        if status != 'Optimal':
            break
        elif var[key].varValue>0:
            RAND_RECOMMAND.loc[RAND_RECOMMAND_INDEX]=[ITEM_CATEGORY.loc[key,'ITEM_CATEGORY'],ITEM_CATEGORY.loc[key,'ITEM'],ITEM_CATEGORY.loc[key,'ITEM_DESCRIPTION'],var[key].varValue,ITEM_CATEGORY.loc[key,'PRICE']]
            RAND_RECOMMAND_INDEX += 1
        else:
            pass
    RAND_RECOMMAND=RAND_RECOMMAND.sort_values(by=['ITEM_CATEGORY','ITEM'])
    return RAND_RECOMMAND
{% endcodeblock %}
可以看到，虽然scipy也可以求解线性规划问题，但是使用Pulp的话整个过程直接写式子就好，理解起来会更简单，也更容易修改方便模型变化。
此外我们一共需要提供五个接口给前台，分别是查询所有的主食、价格、选定主食时查价格、选定价格时查主食，还有最后推荐计算。先展示路由：
{% codeblock lang:python %}
from django.urls import path,re_path
from . import views
app_name = 'APP1'
urlpatterns = [
    re_path(r'PriceSelected(\d+)/',views.PriceSelected),
    re_path(r'StapleSelected(\d+)/',views.StapleSelected),
    path(r'Recommend/',views.Recommend),
    path(r'Price',views.Price),
    path(r'Staple/',views.Staple),
    path('', views.Index, name='index'),
]
{% endcodeblock %}
推荐的部分重点内容前面已经讲到了，这里主要说其他四个接口，对应的业务处理views.py部分的方法：
{% codeblock lang:python %}
def Staple(request):
    # Give all staple directly for choose
    Staple = ITEM_CATEGORY[ITEM_CATEGORY['ITEM_CATEGORY']=='Staple'] 
    list = []
    for index in Staple.index:
        list.append({'id':str(index),'name':str(Staple.loc[index,'ITEM_DESCRIPTION'])})   
    return JsonResponse({'data':list})

def Price(request):
    # Give all price.dropduplicates directly for choose
    Price = pd.DataFrame(ITEM_CATEGORY[ITEM_CATEGORY['ITEM_CATEGORY']=='Staple']['PRICE'].drop_duplicates().sort_values())
    list = []
    for index in Price.index:
        list.append({'id':str(index),'name':str(Price.loc[index,'PRICE'])})
    return JsonResponse({'data':list})

def StapleSelected(request,id):
    # When cilcking on a staple, get the price
    Price = ITEM_CATEGORY.loc[int(id),'PRICE']
    list = []
    list.append({'id':str(Price),'name':str(Price)})
    return JsonResponse({'data':list})

def PriceSelected(request,id):
    # When cilcking on a price, get the stape
    Price = ITEM_CATEGORY.loc[int(id),'PRICE']
    Staple = ITEM_CATEGORY[(ITEM_CATEGORY['ITEM_CATEGORY']=='Staple') & (ITEM_CATEGORY['PRICE']==Price)]
    list = []
    for index in Staple.index:
        list.append({'id':str(index),'name':str(Staple.loc[index,'ITEM_DESCRIPTION'])})  
    return JsonResponse({'data':list})
{% endcodeblock %}
对应的前台处理html中js部分的方法：
{% codeblock lang:python %}
 // Get all staples
$.get('/APP1/Staple',function (dic) {
    $.each(dic.data,function (index, item) {
        $('#Staple').append('<option value="'+item.id+'">'+item.name+'</option>')
    })
})

// Get all stapleprices
$.get('/APP1/Price',function (dic) {
    $.each(dic.data,function (index, item) {
        $('#StaplePrice').append('<option value="'+item.id+'">'+item.name+'</option>')
    })
})

// When staple selected, change the price
$('#Staple').change(function () {
    if(parseInt($(this).val()) >= 0){
        $.get('/APP1/StapleSelected'+$(this).val()+'/',function (dic) {
            $.each(dic.data,function (index, item) {
                $('#StaplePrice').empty().append('<option value="'+item.id+'">'+item.name+'</option>')
            })
            $('#StaplePrice').setAttribute('disabled','true');
        })
    }else{
        ajaxInfo();
    }
})

// When price selected, change the staple
$('#StaplePrice').change(function () {
    $('#Staple').empty().append('<option> choose staple </option>')
    if(parseInt($(this).val()) >= 0){
        $.get('/APP1/PriceSelected'+$(this).val()+'/',function (dic) {
            $.each(dic.data,function (index, item) {
                $('#Staple').append('<option value="'+item.id+'">'+item.name+'</option>')
            })
        })
    }else{
        ajaxInfo();
    }
})
{% endcodeblock %}
&emsp;&emsp;OK，进入页面，输入总的支付意愿，开始生成推荐吧，不喜欢？再点一次~
&emsp;&emsp;完整的内容还是直接到项目里去看吧应该会更容易理解。