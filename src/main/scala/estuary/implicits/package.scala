package estuary

import breeze.linalg.DenseMatrix
import estuary.utils.RichMatrix

/**
  * Created by mengpan on 2017/10/27.
  */
package object implicits {

  implicit def enrichMatrix(m: DenseMatrix[Double]): RichMatrix = {
    RichMatrix(m.data, m.rows, m.cols)
  }
}
