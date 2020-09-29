import tensorflow as tf
import pprint
import os
import numpy as np

  
def tf2keras(model):
  pp = pprint.PrettyPrinter(indent=4)
  weight_path = "/media/yiling/DC30502330500744/yiling/intel2020/dataset/checkpoints/SemanticKITTI-20200904T143432Z-001/SemanticKITTI/"
  # Retrieve weights from TF checkpoint
  tf_path = os.path.abspath(weight_path)
  init_vars = tf.train.list_variables(tf_path)
  tf_vars = []
  for name, shape in init_vars:
      #print("Loading TF weight {} with shape {}".format(name, shape))
      array = tf.train.load_variable(tf_path, name)
      tf_vars.append((name, array.squeeze()))

  # FOr each variable in the PyTorch model
  for name, array in tf_vars:
      # skip the prefix ('model/') and split the path-like variable name in a list of sub-path
      
     
      whole_name = name
      name = name.split('/')

      if (name[0]=="optimizer"):
          continue
      name = name[1:]


      # Initiate the pointer from the main model class
      pointer = model

      # We iterate along the scopes and move our pointer accordingly
      for m_name in name:

          l = [m_name]

          # Convert parameters final names to the PyTorch modules equivalent names
          last_pointer = pointer
          if l[0] == 'beta':
              pointer = getattr(pointer, 'beta')
          elif l[0] == 'gamma':
              pointer = getattr(pointer, 'gamma')
          elif l[0] == 'moving_mean':
              pointer = getattr(pointer, 'moving_mean')
          elif l[0] == 'moving_variance':
              pointer = getattr(pointer, 'moving_variance')
          elif l[0] == 'kernel':
              pointer = getattr(pointer, 'kernel')
          elif l[0] == 'w' or l[0] == 'g':
              pointer = getattr(pointer, 'kernel')
          elif l[0] == 'b':
              pointer = getattr(pointer, 'bias')
          elif l[0] == 'biases':
              pointer = getattr(pointer, 'conv')
              pointer = getattr(pointer, 'bias')
          elif l[0] == 'weights':
              pointer = getattr(pointer, 'conv')
              pointer = getattr(pointer, 'kernel')
          # elif l[0] == 'biases':
          #     pointer = getattr(pointer, 'conv')
          #     pointer = getattr(pointer, 'bias')
          else:
              #pprint(vars(pointer))
              # print("==================")
              # print(name)
              # pp.pprint(vars(pointer))

              pointer = getattr(pointer, l[0])
   

       
      # if (l[0]=='kernel'):
      #     array = np.transpose(array)

      if (len(pointer.shape)==4 and pointer.shape[0]==pointer.shape[1] and pointer.shape[1]==1):
          # array = np.transpose(array)
          array = np.expand_dims(array, axis=0)
          array = np.expand_dims(array, axis=0)
     

      try:
          # print("----")
          # print(pointer.shape)
          # print(array.shape)
          # # for t in pointer:
          # #   print(t)
          # #   print("==")
          # print(array.shape)  
          assert pointer.shape == array.shape  # Catch error if the array shapes are not identical
      except AssertionError as e:
          # print(whole_name)
          #pprint(vars(last_pointer))
          #print(last_pointer.weights.shape)
          e.args += (pointer.shape, array.shape)
          raise

      #print("Initialize PyTorch weight {}".format(name))
      # print(pointer)
      # print(pointer.numpy)
      if whole_name == 'layers/Encoder_layer_0LFAatt_pooling_1mlp/weights':
          pointer.assign(array)
          print(pointer.numpy)
          print(array)
          print("====")
      pointer.assign(array)
  print("Initialize torch weights from {}".format(weight_path))