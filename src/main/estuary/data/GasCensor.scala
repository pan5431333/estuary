package estuary.data

import breeze.linalg.DenseVector

/**
  * Created by mengpan on 2017/8/24.
  */
case class GasCensor(var feature: DenseVector[Double], var label: Double) extends Data

