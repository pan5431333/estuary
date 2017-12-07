package estuary.data

import breeze.linalg.DenseMatrix
import estuary.support.RichMatrix

import scala.util.Random
import estuary.implicits._
import org.slf4j.LoggerFactory

/**
  * Created by mengpan on 2017/10/27.
  */
class GasCensorDataReader extends Reader with Serializable{
  val log = LoggerFactory.getLogger(this.getClass)

  override def read(filePath: String): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val fileNames = getMatchedFilesName(filePath)
    log.info("Matched Files: ")
    fileNames foreach log.info
    val dataFile = fileNames.filter(_.toLowerCase.contains("feature")).head
    val labelFile = fileNames.filter(_.toLowerCase.contains("label")).head
    val data = RichMatrix.read(dataFile).toDenseMatrix
    val label = RichMatrix.read(labelFile).toDenseMatrix
    (data, label)
  }
}

object GasCensorDataReader extends App{
  new GasCensorDataReader().partition("""D:\\Users\\m_pan\\Downloads\\Dataset\\Dataset\\train.*""", 4)
}
