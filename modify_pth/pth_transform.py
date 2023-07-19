import torch
from collections import OrderedDict

pthfile = './epoch_300_modify.pth'    # 需要修改的.pth文件路径
model = torch.load(pthfile, torch.device('cuda'))   # 读取需要修改的.pth文件到model
# my_dict的key，对应修改后的层名称。value对应需要进行修改的原始名称。
my_dict = {'cls_token': 'backbone.cls_token',
           'pos_embed': 'backbone.pos_embed',
           'patch_embed.proj.weight': 'backbone.patch_embed.projection.weight',
           'patch_embed.proj.bias': 'backbone.patch_embed.projection.bias',

           'blocks.0.norm1.weight': 'backbone.layers.0.ln1.weight',
           'blocks.0.norm1.bias': 'backbone.layers.0.ln1.bias',
           'blocks.0.attn.qkv.weight': 'backbone.layers.0.attn.qkv.weight',
           'blocks.0.attn.proj.weight': 'backbone.layers.0.attn.proj.weight',
           'blocks.0.attn.proj.bias': 'backbone.layers.0.attn.proj.bias',
           'blocks.0.norm2.weight': 'backbone.layers.0.ln2.weight',
           'blocks.0.norm2.bias': 'backbone.layers.0.ln2.bias',
           'blocks.0.mlp.fc1.weight': 'backbone.layers.0.ffn.layers.0.0.weight',
           'blocks.0.mlp.fc1.bias': 'backbone.layers.0.ffn.layers.0.0.bias',
           'blocks.0.mlp.fc2.weight': 'backbone.layers.0.ffn.layers.1.weight',
           'blocks.0.mlp.fc2.bias': 'backbone.layers.0.ffn.layers.1.bias',

           'blocks.1.norm1.weight': 'backbone.layers.1.ln1.weight',
           'blocks.1.norm1.bias': 'backbone.layers.1.ln1.bias',
           'blocks.1.attn.qkv.weight': 'backbone.layers.1.attn.qkv.weight',
           'blocks.1.attn.proj.weight': 'backbone.layers.1.attn.proj.weight',
           'blocks.1.attn.proj.bias': 'backbone.layers.1.attn.proj.bias',
           'blocks.1.norm2.weight': 'backbone.layers.1.ln2.weight',
           'blocks.1.norm2.bias': 'backbone.layers.1.ln2.bias',
           'blocks.1.mlp.fc1.weight': 'backbone.layers.1.ffn.layers.0.0.weight',
           'blocks.1.mlp.fc1.bias': 'backbone.layers.1.ffn.layers.0.0.bias',
           'blocks.1.mlp.fc2.weight': 'backbone.layers.1.ffn.layers.1.weight',
           'blocks.1.mlp.fc2.bias': 'backbone.layers.1.ffn.layers.1.bias',

           'blocks.2.norm1.weight': 'backbone.layers.2.ln1.weight',
           'blocks.2.norm1.bias': 'backbone.layers.2.ln1.bias',
           'blocks.2.attn.qkv.weight': 'backbone.layers.2.attn.qkv.weight',
           'blocks.2.attn.proj.weight': 'backbone.layers.2.attn.proj.weight',
           'blocks.2.attn.proj.bias': 'backbone.layers.2.attn.proj.bias',
           'blocks.2.norm2.weight': 'backbone.layers.2.ln2.weight',
           'blocks.2.norm2.bias': 'backbone.layers.2.ln2.bias',
           'blocks.2.mlp.fc1.weight': 'backbone.layers.2.ffn.layers.0.0.weight',
           'blocks.2.mlp.fc1.bias': 'backbone.layers.2.ffn.layers.0.0.bias',
           'blocks.2.mlp.fc2.weight': 'backbone.layers.2.ffn.layers.1.weight',
           'blocks.2.mlp.fc2.bias': 'backbone.layers.2.ffn.layers.1.bias',

           'blocks.3.norm1.weight': 'backbone.layers.3.ln1.weight',
           'blocks.3.norm1.bias': 'backbone.layers.3.ln1.bias',
           'blocks.3.attn.qkv.weight': 'backbone.layers.3.attn.qkv.weight',
           'blocks.3.attn.proj.weight': 'backbone.layers.3.attn.proj.weight',
           'blocks.3.attn.proj.bias': 'backbone.layers.3.attn.proj.bias',
           'blocks.3.norm2.weight': 'backbone.layers.3.ln2.weight',
           'blocks.3.norm2.bias': 'backbone.layers.3.ln2.bias',
           'blocks.3.mlp.fc1.weight': 'backbone.layers.3.ffn.layers.0.0.weight',
           'blocks.3.mlp.fc1.bias': 'backbone.layers.3.ffn.layers.0.0.bias',
           'blocks.3.mlp.fc2.weight': 'backbone.layers.3.ffn.layers.1.weight',
           'blocks.3.mlp.fc2.bias': 'backbone.layers.3.ffn.layers.1.bias',

           'blocks.4.norm1.weight': 'backbone.layers.4.ln1.weight',
           'blocks.4.norm1.bias': 'backbone.layers.4.ln1.bias',
           'blocks.4.attn.qkv.weight': 'backbone.layers.4.attn.qkv.weight',
           'blocks.4.attn.proj.weight': 'backbone.layers.4.attn.proj.weight',
           'blocks.4.attn.proj.bias': 'backbone.layers.4.attn.proj.bias',
           'blocks.4.norm2.weight': 'backbone.layers.4.ln2.weight',
           'blocks.4.norm2.bias': 'backbone.layers.4.ln2.bias',
           'blocks.4.mlp.fc1.weight': 'backbone.layers.4.ffn.layers.0.0.weight',
           'blocks.4.mlp.fc1.bias': 'backbone.layers.4.ffn.layers.0.0.bias',
           'blocks.4.mlp.fc2.weight': 'backbone.layers.4.ffn.layers.1.weight',
           'blocks.4.mlp.fc2.bias': 'backbone.layers.4.ffn.layers.1.bias',

           'blocks.5.norm1.weight': 'backbone.layers.5.ln1.weight',
           'blocks.5.norm1.bias': 'backbone.layers.5.ln1.bias',
           'blocks.5.attn.qkv.weight': 'backbone.layers.5.attn.qkv.weight',
           'blocks.5.attn.proj.weight': 'backbone.layers.5.attn.proj.weight',
           'blocks.5.attn.proj.bias': 'backbone.layers.5.attn.proj.bias',
           'blocks.5.norm2.weight': 'backbone.layers.5.ln2.weight',
           'blocks.5.norm2.bias': 'backbone.layers.5.ln2.bias',
           'blocks.5.mlp.fc1.weight': 'backbone.layers.5.ffn.layers.0.0.weight',
           'blocks.5.mlp.fc1.bias': 'backbone.layers.5.ffn.layers.0.0.bias',
           'blocks.5.mlp.fc2.weight': 'backbone.layers.5.ffn.layers.1.weight',
           'blocks.5.mlp.fc2.bias': 'backbone.layers.5.ffn.layers.1.bias',

           'blocks.6.norm1.weight': 'backbone.layers.6.ln1.weight',
           'blocks.6.norm1.bias': 'backbone.layers.6.ln1.bias',
           'blocks.6.attn.qkv.weight': 'backbone.layers.6.attn.qkv.weight',
           'blocks.6.attn.proj.weight': 'backbone.layers.6.attn.proj.weight',
           'blocks.6.attn.proj.bias': 'backbone.layers.6.attn.proj.bias',
           'blocks.6.norm2.weight': 'backbone.layers.6.ln2.weight',
           'blocks.6.norm2.bias': 'backbone.layers.6.ln2.bias',
           'blocks.6.mlp.fc1.weight': 'backbone.layers.6.ffn.layers.0.0.weight',
           'blocks.6.mlp.fc1.bias': 'backbone.layers.6.ffn.layers.0.0.bias',
           'blocks.6.mlp.fc2.weight': 'backbone.layers.6.ffn.layers.1.weight',
           'blocks.6.mlp.fc2.bias': 'backbone.layers.6.ffn.layers.1.bias',

           'blocks.7.norm1.weight': 'backbone.layers.7.ln1.weight',
           'blocks.7.norm1.bias': 'backbone.layers.7.ln1.bias',
           'blocks.7.attn.qkv.weight': 'backbone.layers.7.attn.qkv.weight',
           'blocks.7.attn.proj.weight': 'backbone.layers.7.attn.proj.weight',
           'blocks.7.attn.proj.bias': 'backbone.layers.7.attn.proj.bias',
           'blocks.7.norm2.weight': 'backbone.layers.7.ln2.weight',
           'blocks.7.norm2.bias': 'backbone.layers.7.ln2.bias',
           'blocks.7.mlp.fc1.weight': 'backbone.layers.7.ffn.layers.0.0.weight',
           'blocks.7.mlp.fc1.bias': 'backbone.layers.7.ffn.layers.0.0.bias',
           'blocks.7.mlp.fc2.weight': 'backbone.layers.7.ffn.layers.1.weight',
           'blocks.7.mlp.fc2.bias': 'backbone.layers.7.ffn.layers.1.bias',

           'blocks.8.norm1.weight': 'backbone.layers.8.ln1.weight',
           'blocks.8.norm1.bias': 'backbone.layers.8.ln1.bias',
           'blocks.8.attn.qkv.weight': 'backbone.layers.8.attn.qkv.weight',
           'blocks.8.attn.proj.weight': 'backbone.layers.8.attn.proj.weight',
           'blocks.8.attn.proj.bias': 'backbone.layers.8.attn.proj.bias',
           'blocks.8.norm2.weight': 'backbone.layers.8.ln2.weight',
           'blocks.8.norm2.bias': 'backbone.layers.8.ln2.bias',
           'blocks.8.mlp.fc1.weight': 'backbone.layers.8.ffn.layers.0.0.weight',
           'blocks.8.mlp.fc1.bias': 'backbone.layers.8.ffn.layers.0.0.bias',
           'blocks.8.mlp.fc2.weight': 'backbone.layers.8.ffn.layers.1.weight',
           'blocks.8.mlp.fc2.bias': 'backbone.layers.8.ffn.layers.1.bias',

           'blocks.9.norm1.weight': 'backbone.layers.9.ln1.weight',
           'blocks.9.norm1.bias': 'backbone.layers.9.ln1.bias',
           'blocks.9.attn.qkv.weight': 'backbone.layers.9.attn.qkv.weight',
           'blocks.9.attn.proj.weight': 'backbone.layers.9.attn.proj.weight',
           'blocks.9.attn.proj.bias': 'backbone.layers.9.attn.proj.bias',
           'blocks.9.norm2.weight': 'backbone.layers.9.ln2.weight',
           'blocks.9.norm2.bias': 'backbone.layers.9.ln2.bias',
           'blocks.9.mlp.fc1.weight': 'backbone.layers.9.ffn.layers.0.0.weight',
           'blocks.9.mlp.fc1.bias': 'backbone.layers.9.ffn.layers.0.0.bias',
           'blocks.9.mlp.fc2.weight': 'backbone.layers.9.ffn.layers.1.weight',
           'blocks.9.mlp.fc2.bias': 'backbone.layers.9.ffn.layers.1.bias',

           'blocks.10.norm1.weight': 'backbone.layers.10.ln1.weight',
           'blocks.10.norm1.bias': 'backbone.layers.10.ln1.bias',
           'blocks.10.attn.qkv.weight': 'backbone.layers.10.attn.qkv.weight',
           'blocks.10.attn.proj.weight': 'backbone.layers.10.attn.proj.weight',
           'blocks.10.attn.proj.bias': 'backbone.layers.10.attn.proj.bias',
           'blocks.10.norm2.weight': 'backbone.layers.10.ln2.weight',
           'blocks.10.norm2.bias': 'backbone.layers.10.ln2.bias',
           'blocks.10.mlp.fc1.weight': 'backbone.layers.10.ffn.layers.0.0.weight',
           'blocks.10.mlp.fc1.bias': 'backbone.layers.10.ffn.layers.0.0.bias',
           'blocks.10.mlp.fc2.weight': 'backbone.layers.10.ffn.layers.1.weight',
           'blocks.10.mlp.fc2.bias': 'backbone.layers.10.ffn.layers.1.bias',

           'blocks.11.norm1.weight': 'backbone.layers.11.ln1.weight',
           'blocks.11.norm1.bias': 'backbone.layers.11.ln1.bias',
           'blocks.11.attn.qkv.weight': 'backbone.layers.11.attn.qkv.weight',
           'blocks.11.attn.proj.weight': 'backbone.layers.11.attn.proj.weight',
           'blocks.11.attn.proj.bias': 'backbone.layers.11.attn.proj.bias',
           'blocks.11.norm2.weight': 'backbone.layers.11.ln2.weight',
           'blocks.11.norm2.bias': 'backbone.layers.11.ln2.bias',
           'blocks.11.mlp.fc1.weight': 'backbone.layers.11.ffn.layers.0.0.weight',
           'blocks.11.mlp.fc1.bias': 'backbone.layers.11.ffn.layers.0.0.bias',
           'blocks.11.mlp.fc2.weight': 'backbone.layers.11.ffn.layers.1.weight',
           'blocks.11.mlp.fc2.bias': 'backbone.layers.11.ffn.layers.1.bias',

           'norm.weight': 'backbone.ln1.weight',
           'norm.bias': 'backbone.ln1.bias',
           'blocks.0.attn.qkv.bias': 'backbone.layers.0.attn.qkv.bias',
           'blocks.1.attn.qkv.bias': 'backbone.layers.1.attn.qkv.bias',
           'blocks.2.attn.qkv.bias': 'backbone.layers.2.attn.qkv.bias',
           'blocks.3.attn.qkv.bias': 'backbone.layers.3.attn.qkv.bias',
           'blocks.4.attn.qkv.bias': 'backbone.layers.4.attn.qkv.bias',
           'blocks.5.attn.qkv.bias': 'backbone.layers.5.attn.qkv.bias',
           'blocks.6.attn.qkv.bias': 'backbone.layers.6.attn.qkv.bias',
           'blocks.7.attn.qkv.bias': 'backbone.layers.7.attn.qkv.bias',
           'blocks.8.attn.qkv.bias': 'backbone.layers.8.attn.qkv.bias',
           'blocks.9.attn.qkv.bias': 'backbone.layers.9.attn.qkv.bias',
           'blocks.10.attn.qkv.bias': 'backbone.layers.10.attn.qkv.bias',
           'blocks.11.attn.qkv.bias': 'backbone.layers.11.attn.qkv.bias'
           }
# 由于使用value，不便于对key进行检索，交换my_dict字典的键值对。
old2new_dic = {v: k for k, v in my_dict.items()}


# 读取.pth文件，并打印其类型（通常是dict）、打印其字典长度、返回其key的值
def read_model(model):
    print('type: ', type(model), '\n**************************************')
    print('length:', len(model), '\n**************************************')  # 查看模型字典长度, 长度为 n ，即存在n个 key-value 键值对
    keys = []
    for k in model.keys():  # 查看模型字典里面的key
        keys.append(k)
    return keys


# 针对使用mmpretrain预训练出来的.pth文件，model中会有{meta, state_dict,
# message_hub, optimizer, param_schedulers}5个key，我们只保留state_dict及其对应的值。
def remove_keys(model):
    del model['meta']
    del model['message_hub']
    del model['optimizer']
    del model['param_schedulers']
    model['model'] = model.pop('state_dict')
    return model


# 读取model['model']中包含的层，以及层大小。打印出来，并返回层名称列表。
def read_model_layers(model):
    l = model['model']
    for k, v in l.items():
        print(k, v.size(), sep='  -->  ')
    old_name_list = list(l.keys())
    return old_name_list


# 最关键的函数，修改函数：输入旧模型和字典。返回新模型
# new_model = OrderedDict()
def change_layers_names(old_model, old2new_dic):
    renamed_model = OrderedDict()
    model = remove_keys(old_model)
    old_name_list = read_model_layers(model)
    l = model['model']
    for old_name in old_name_list:  #
        for key in old2new_dic.keys():
            if old_name == key:
                renamed_model[old2new_dic[key]] = l[old_name]
            else:
                pass
    renamed_model = {'model': renamed_model}
    print(renamed_model)
    torch.save(renamed_model, './renamed_model.pth')