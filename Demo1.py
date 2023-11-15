import numpy as np

# 激活函数
def activation_relu(inputs):
    return np.maximum(0,inputs) 
# softmax激活函数
def activation_softmax(inputs):
    max_values = np.max(inputs,axis=1,keepdims=True) #保持之前的维度
    slided_inputs= inputs - max_values    #在之后的指数计算中比值仍不变，将指数变负可防止指数爆炸
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values,axis=1,keepdims=True)
    norm_values = exp_values/norm_base
    return norm_values

Network=[2,4,3,2]
inputs=np.array([[0.5,-0.2],
                 [0.3,0.4],
                 [-0.5,0.6],
                 [0.7,-0.8],
                 [0.9,0.1]])

# 定义一个层类
class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.biases=np.random.randn(n_neurons)
    def forward(self,inputs):
        sum1=np.dot(inputs,self.weights)+self.biases
        self.outputs=activation_relu(sum1)
        return self.outputs

# 定义一个网络类
class net:
    def __init__(self,Network):
        self.shape=Network
        self.layers=[]  #新建一个层类数组
        for i in range(len(Network)-1):
            nlayer=layer(Network[i],Network[i+1])
            self.layers.append(nlayer)
    
    def creat(self,inputs):
        outputs=[inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].forward(outputs[i])
            if i < len(self.layers)-1:
                layer_output= activation_relu(layer_sum)
            else:
                layer_output= activation_softmax(layer_sum)
            outputs.append(layer_output)#这里选择把每一层的结果都保留在数组中，以备用  
        print (outputs)
        return outputs
def main():
    net0=net(Network)
    net0.creat(inputs)

main()