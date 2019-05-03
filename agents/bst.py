'''
Fixed size binary search tree implementation
'''

import numpy as np
from collections import deque
from collections import namedtuple


class Node(object):
	def __init__(self,data):
		self.data = data
		self.freq = 1

		self.leftchild = None
		self.rightchild = None

	def assign(self,another_node):
		self.data = another_node.data
		self.freq = another_node.freq

		self.leftchild = another_node.leftchild
		self.rightchild = another_node.rightchild

	def remove_child(self,child_node):
		assert child_node == self.leftchild or child_node == self.rightchild

		if self.data < child_node.data:
			self.rightchild = None
		else:
			self.leftchild = None


class FixedSize_BinarySearchTree(object):
	def __init__(self,capacity):
		
		self.capacity = capacity
		self.size = 0

		self.values = deque(maxlen=capacity)
		#self.value_sum = 0

		self.root = None

	
	def update(self,value,idx,node=None):
		'''
		update tree node value
		'''

		assert idx < self.size

		self.remove(self.values[idx])

		self.insert(value)

		#self.value_sum += value - self.values[idx]
		self.values[idx] = value
		self.size+=1

		# try:
		# 	assert abs(np.sum(self.values,dtype=np.float32)-self.value_sum)<1e-3
		# except:
		# 	print (abs(np.sum(self.values,dtype=np.float32)-self.value_sum))
		# 	raise Exception('inconsistent values')
	
	def insert(self,value,node=None):
		'''
		update tree node value
		'''

		if self.root is None:
			self.root = Node(value)
			assert(len(self.values) == 0)
		
		elif node is None:
			return self.insert(value,node=self.root)

		elif value == node.data:
			node.freq += 1

		elif value > node.data:
			if node.rightchild:
				return self.insert(value,node=node.rightchild)
			else:
				node.rightchild = Node(value)
				
		else:
			if node.leftchild:
				return self.insert(value,node.leftchild)	
			else:
				node.leftchild = Node(value)
	
	def add(self,value,node=None):
		'''
		add tree node
		'''

		if self.size == self.capacity:
			self.remove(self.values[0])
			#self.value_sum -= self.values[0]

		self.insert(value)

		#self.value_sum += value
		self.values.append(value)
		self.size+=1

		# try:
		# 	assert abs(np.sum(self.values,dtype=np.float32)-self.value_sum)<1e-3
		# except:
		# 	print (abs(np.sum(self.values,dtype=np.float32)-self.value_sum))
		# 	raise Exception('inconsistent values')
	
	def search(self,value):
		'''
		search for node with a particular value in the tree
		'''
		parent_node = None
		node = self.root

		while node is not None and value!=node.data:
			parent_node = node

			if value > node.data:
				node = node.rightchild

			else:
				node = node.leftchild

		return parent_node,node


	def RightMinChild(self,node):
		'''
		get min value subchild for a node
		'''

		assert node.rightchild,"there is no right child for the given node"

		parent_node = node
		node = node.rightchild

		while node.leftchild:
			parent_node = node
			node = node.leftchild

		return parent_node,node


	def remove(self,value):
		'''
		remove tree node
		'''

		parent_node,node = self.search(value)

		if node is None:
			raise Exception('binary search tree has no node with value: {}'.format(value))

		elif node.freq >=2 :
			node.freq -= 1

		elif node.rightchild is None and node.leftchild is None:
			if parent_node:
				parent_node.remove_child(node)
			else:
				self.root = None

		elif node.rightchild is None:
			node.assign(node.leftchild)

		elif node.leftchild is None:
			node.assign(node.rightchild)

		else:
			parent_min_node,min_value_node = self.RightMinChild(node)
			temp_data,temp_freq = min_value_node.data,min_value_node.freq

			if min_value_node.rightchild:
				min_value_node.assign(min_value_node.rightchild)
			else:
				parent_min_node.remove_child(min_value_node)

			node.data = temp_data
			node.freq = temp_freq

		self.size -=1

	def max_value(self):
		assert (self.size != 0)

		node = self.root
		while node.rightchild is not None:
			node = node.rightchild

		return node.data


	def __len__(self):
		return self.size

