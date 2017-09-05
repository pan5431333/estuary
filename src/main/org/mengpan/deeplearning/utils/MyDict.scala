package org.mengpan.deeplearning.utils

/**
  * Created by mengpan on 2017/8/23.
  */
object MyDict {

  //设定激活函数
  //0 sigmoid, 1 tanh, 2 ReLU, 3 Leaky ReLU
  val ACTIVATION_SIGMOID: Byte = 0
  val ACTIVATION_TANH: Byte = 1
  val ACTIVATION_RELU: Byte = 2
  val ACTIVATION_LEAKY_RELU: Byte = 3

  //设定权重初始化的方法
  //0 He, 1 Xaiver
  val INIT_HE: Byte = 0
  val INIT_XAIVER: Byte = 1

  //设定正则化方法
  //0 L2, 1 L1
  val REGULARIZATION_L2: Byte = 0
  val REGULARIZATION_L1: Byte = 1
}
