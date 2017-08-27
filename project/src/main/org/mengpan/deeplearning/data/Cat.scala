package org.mengpan.deeplearning.data

import breeze.linalg.DenseVector

/**
  * Created by mengpan on 2017/8/15.
  */
case class Cat(var feature: DenseVector[Double], var label: Double) extends Data

