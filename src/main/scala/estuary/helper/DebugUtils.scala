package estuary.helper

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by mengpan on 2017/8/26.
  */
object DebugUtils {
  def matrixShape(w: DenseMatrix[Double], objectName: String): String = {
    objectName + "'s shape: (" + w.rows + ", " + w.cols + ")"
  }

  def vectorShape(b: DenseVector[Double], objectName: String): String = {
    objectName + "'s shape: (" + b.length + ")"
  }


}
