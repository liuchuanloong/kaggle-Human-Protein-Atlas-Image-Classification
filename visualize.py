from graphviz import Digraph
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def make_dot(var, params=None):
    """
    画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
    蓝色节点表示有梯度计算的变量Variables;
    橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled', shape='box', align='left',
                              fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr)
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # 多输出场景 multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    # resize_graph(dot)

    return dot

# from RFB_Net_mobile import *
# from RFBnet_vgg import *
x = Variable(torch.randn(2,4,256,256))#change 12 to the channel number of network input
# model = UNetResNet34_DPAN_Hyper()
# model = model = UNetResNet34_MsFPANet_Hyper()
# from DenseNet import DenseNet
# model = DenseNet(growthRate=12, depth=100, reduction=0.5,
#                             bottleneck=True, nClasses=10)
size = 300
# model = RFBNet('train', size, *multibox(size, MobileNet(),
#                                 add_extras(size, extras[str(size)], 1024),
#                                 mbox[str(size)], 2), 2)
# model = RFBNet('train', size, *multibox(size, vgg(base[str(size)], 3),
#                               add_extras(size, extras[str(size)], 1024),
#                               mbox[str(size)], 2), 2)
from Model_c import model50A_RFClass
model = model50A_RFClass()
y = model(x)

for i in model.named_parameters():
    print(i[0],i[1].size())

# g = make_dot(y, params=dict(model.named_parameters()))
# g.view()
with SummaryWriter(comment='Net') as w:
    w.add_graph(model, (x, ))
