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

  def partition(n: Int): Unit = {
    val (feature, label) = read("/Users/mengpan/Downloads/NewDataset/0.*")
    val numExamples = feature.rows
    val shuffledIndex = Random.shuffle[Int, Vector]((0 until numExamples).toVector)
    val nn = numExamples

    for (i <- 0 until n) yield {
      feature(shuffledIndex.slice(i * nn / n, (i + 1) * nn / n), ::).toDenseMatrix.save(s"/Users/mengpan/Downloads/NewDataset/${i+1}-feature.dat")
      label(shuffledIndex.slice(i * nn / n, (i + 1) * nn / n), ::).toDenseMatrix.save(s"/Users/mengpan/Downloads/NewDataset/${i+1}-label.dat")
    }
  }
}

object GasCensorDataReader extends App{
  new GasCensorDataReader().partition(10)
}
