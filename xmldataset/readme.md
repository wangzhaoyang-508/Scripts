# 文档说明

对使用labelimg标注好的文件进行处理：

**需要**：.xml与.jpg文件在同一个文件夹内



> 功能1：统计文件夹内的文件数量，读取xml文件中的类别和各个样本的个数。
> ```python
> python xml_main.py your/files/path
> ```



> 功能2：检查文件夹内xml文件和对应的图片名称是否对应，将不对应的文件移出至"./path/save/abnormal"
> ```python
> python xml_main.py your/files/path --check_xml ./path/save/abnormal
> ```




>功能3：将xml文件转化为标准的voc格式训练集，手动输入训练集比率（默认test=val）
>默认生成在./voc2012/路径下
>
>```python
>python xml_main.py your/files/path --xml2voc=0.7
>```




>功能4：将功能3生成的voc格式数据集转化为yolo格式数据集
>```python
> python xml_main.py your/files/path --voc2yolo ./path/save/yolo
> ```




>功能5：将功能3生成的voc格式数据集转化为coco格式数据集
>```python
> python xml_main.py your/files/path --voc2coco ./path/save/coco
>```

