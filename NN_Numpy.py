from numpy import exp, array, random, dot,tanh
#from numpy.linalg import inv
import numpy
class neural_network:
	def __init__(self,size1,size2):
		random.seed(1)
		self.hidden_layer_weights1=2*random.random(size1)-1
		self.hidden_layer_weights2=2*random.random(size2)-1
		return 
	def _activation(self,x):
		return tanh(x)
	def _activation_derivative(self,x):
		return 1-x**2
	def _Output(self,Xinput,weights):
		return self._activation(dot(Xinput, weights))
	def train(self, inputX,outputY,no_of_iteration):
		for i in range(no_of_iteration):
			a1=self._Output(inputX,self.hidden_layer_weights1[0,:])
			a2=self._Output(a1,self.hidden_layer_weights1[1,:])
			output=self._Output(a2,self.hidden_layer_weights2)
			error=outputY-output
			adjustment1 = dot(a2.T, error * self._activation_derivative(output))
			error1=dot(output,adjustment1.T)
			adjustment2=  dot(a1.T, error1 * self._activation_derivative(a2))
			error2=dot(a2,adjustment2)
			#print error2		#printing final error. after each iiteration
			adjustment3=  dot(inputX.T, error2 * self._activation_derivative(a1))
			print adjustment1,"\n","_________________________________________________________________________________","\n"
			self.hidden_layer_weights2=adjustment1+self.hidden_layer_weights2
			self.hidden_layer_weights1[1,:]=adjustment2+self.hidden_layer_weights1[1,:]
			self.hidden_layer_weights1[0,:]=adjustment3+self.hidden_layer_weights1[0,:]
			
	def predict(self,inputX):
			a1=self._Output(inputX,self.hidden_layer_weights1[0,:])
			a2=self._Output(a1,self.hidden_layer_weights1[1,:])
			output=self._Output(a2,self.hidden_layer_weights2)
			return output
if __name__=='__main__':
	nn= neural_network((2,3,3),(3,1))
	
	x=array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0]])
	y=array([[0],[1],[1],[0],[1],[0],[0]])
	nn.train(x,y,1000)
	print nn.hidden_layer_weights1[0,:],"\n","\n","\n",nn.hidden_layer_weights1[1,:],"\n","\n","\n", nn.hidden_layer_weights2
	print "\n","________________________________________________________________________________","\n"
	#print nn.predict([[1,1,0]]), "\n"	#predicting the value.
	print nn.hidden_layer_weights1[0,:],"\n","\n","\n",nn.hidden_layer_weights1[1,:],"\n","\n","\n", nn.hidden_layer_weights2
