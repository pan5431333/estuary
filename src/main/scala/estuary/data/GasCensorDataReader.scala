package estuary.data

import breeze.linalg.DenseMatrix
import estuary.utils.RichMatrix

import scala.util.Random
import estuary.implicits._

/**
  * Created by mengpan on 2017/10/27.
  */
class GasCensorDataReader extends Reader with Serializable{
  override def read(filePath: String): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val fileNames = getMatchedFilesName(filePath)
    val dataFile = fileNames.filter(_.endsWith("feature.dat"))(0)
    val labelFile = fileNames.filter(_.endsWith("label.dat"))(0)
    val data = RichMatrix.read(dataFile).toDenseMatrix
    val label = RichMatrix.read(labelFile).toDenseMatrix
    (data, label)
  }


}

object GasCensorDataReader extends App{
  new GasCensorDataReader().partition("/Users/mengpan/Downloads/NewDataset/training.*", 4)
}
