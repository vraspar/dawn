SKIP: FAILED

#version 310 es

struct BST {
  int data;
  int leftIndex;
  int rightIndex;
};

struct main_out {
  vec4 x_GLF_color_1;
};
precision highp float;
precision highp int;


BST tree[10] = BST[10](BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0), BST(0, 0, 0));
vec4 x_GLF_color = vec4(0.0f);
void makeTreeNode_struct_BST_i1_i1_i11_i1_(inout BST node, inout int data) {
  node.data = data;
  node.leftIndex = -1;
  node.rightIndex = -1;
}
void insert_i1_i1_(inout int treeIndex, inout int data_1) {
  int baseIndex = 0;
  BST param = BST(0, 0, 0);
  int param_1 = 0;
  BST param_2 = BST(0, 0, 0);
  int param_3 = 0;
  baseIndex = 0;
  {
    while(true) {
      if ((baseIndex <= treeIndex)) {
      } else {
        break;
      }
      if ((data_1 <= tree[baseIndex].data)) {
        if ((tree[baseIndex].leftIndex == -1)) {
          int x_186 = baseIndex;
          tree[x_186].leftIndex = treeIndex;
          int x_189 = treeIndex;
          param = tree[x_189];
          param_1 = data_1;
          makeTreeNode_struct_BST_i1_i1_i11_i1_(param, param_1);
          tree[x_189] = param;
          return;
        } else {
          baseIndex = tree[baseIndex].leftIndex;
          {
          }
          continue;
        }
      } else {
        if ((tree[baseIndex].rightIndex == -1)) {
          int x_206 = baseIndex;
          tree[x_206].rightIndex = treeIndex;
          int x_209 = treeIndex;
          param_2 = tree[x_209];
          param_3 = data_1;
          makeTreeNode_struct_BST_i1_i1_i11_i1_(param_2, param_3);
          tree[x_209] = param_2;
          return;
        } else {
          baseIndex = tree[baseIndex].rightIndex;
          {
          }
          continue;
        }
      }
      /* unreachable */
    }
  }
}
int search_i1_(inout int t) {
  int index = 0;
  BST currentNode = BST(0, 0, 0);
  int x_220 = 0;
  index = 0;
  {
    while(true) {
      if ((index != -1)) {
      } else {
        break;
      }
      currentNode = tree[index];
      if ((currentNode.data == t)) {
        int x_237 = t;
        return x_237;
      }
      if ((t > currentNode.data)) {
        x_220 = currentNode.rightIndex;
      } else {
        x_220 = currentNode.leftIndex;
      }
      index = x_220;
      {
      }
      continue;
    }
  }
  return -1;
}
void main_1() {
  int treeIndex_1 = 0;
  BST param_4 = BST(0, 0, 0);
  int param_5 = 0;
  int param_6 = 0;
  int param_7 = 0;
  int param_8 = 0;
  int param_9 = 0;
  int param_10 = 0;
  int param_11 = 0;
  int param_12 = 0;
  int param_13 = 0;
  int param_14 = 0;
  int param_15 = 0;
  int param_16 = 0;
  int param_17 = 0;
  int param_18 = 0;
  int param_19 = 0;
  int param_20 = 0;
  int param_21 = 0;
  int param_22 = 0;
  int param_23 = 0;
  int count = 0;
  int i = 0;
  int result = 0;
  int param_24 = 0;
  treeIndex_1 = 0;
  param_4 = tree[0];
  param_5 = 9;
  makeTreeNode_struct_BST_i1_i1_i11_i1_(param_4, param_5);
  tree[0] = param_4;
  treeIndex_1 = (treeIndex_1 + 1);
  param_6 = treeIndex_1;
  param_7 = 5;
  insert_i1_i1_(param_6, param_7);
  treeIndex_1 = (treeIndex_1 + 1);
  param_8 = treeIndex_1;
  param_9 = 12;
  insert_i1_i1_(param_8, param_9);
  treeIndex_1 = (treeIndex_1 + 1);
  param_10 = treeIndex_1;
  param_11 = 15;
  insert_i1_i1_(param_10, param_11);
  treeIndex_1 = (treeIndex_1 + 1);
  param_12 = treeIndex_1;
  param_13 = 7;
  insert_i1_i1_(param_12, param_13);
  treeIndex_1 = (treeIndex_1 + 1);
  param_14 = treeIndex_1;
  param_15 = 8;
  insert_i1_i1_(param_14, param_15);
  treeIndex_1 = (treeIndex_1 + 1);
  param_16 = treeIndex_1;
  param_17 = 2;
  insert_i1_i1_(param_16, param_17);
  treeIndex_1 = (treeIndex_1 + 1);
  param_18 = treeIndex_1;
  param_19 = 6;
  insert_i1_i1_(param_18, param_19);
  treeIndex_1 = (treeIndex_1 + 1);
  param_20 = treeIndex_1;
  param_21 = 17;
  insert_i1_i1_(param_20, param_21);
  treeIndex_1 = (treeIndex_1 + 1);
  param_22 = treeIndex_1;
  param_23 = 13;
  insert_i1_i1_(param_22, param_23);
  count = 0;
  i = 0;
  {
    while(true) {
      if ((i < 20)) {
      } else {
        break;
      }
      param_24 = i;
      int x_132 = search_i1_(param_24);
      result = x_132;
      int x_133 = i;
      switch(x_133) {
        case 2:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 12:
        case 13:
        case 15:
        case 17:
        {
          if ((result == i)) {
            count = (count + 1);
          }
          break;
        }
        default:
        {
          if ((result == -1)) {
            count = (count + 1);
          }
          break;
        }
      }
      {
        i = (i + 1);
      }
      continue;
    }
  }
  if ((count == 20)) {
    x_GLF_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
  } else {
    x_GLF_color = vec4(0.0f, 0.0f, 1.0f, 1.0f);
  }
}
main_out main() {
  main_1();
  return main_out(x_GLF_color);
}
error: Error parsing GLSL shader:
ERROR: 0:10: 'float' : type requires declaration of default precision qualifier 
ERROR: 0:10: '' : compilation terminated 
ERROR: 2 compilation errors.  No code generated.




tint executable returned error: exit status 1
