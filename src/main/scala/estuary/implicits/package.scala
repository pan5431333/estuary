package estuary

import breeze.linalg.DenseMatrix
import estuary.components.layers.ConvLayer.RichImageFeature
import estuary.utils.RichMatrix

/**
  * Created by mengpan on 2017/10/27.
  */
package object implicits {

  implicit def enrichMatrix(m: DenseMatrix[Double]): RichMatrix = {
    RichMatrix(m.data, m.rows, m.cols)
  }

  implicit def convertSeqRichImageFeatureToMatrix(a: Seq[RichImageFeature]): RichMatrix = {
    RichMatrix.create(a)
  }
}
