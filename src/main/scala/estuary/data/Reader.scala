package estuary.data

import java.io.File
import breeze.linalg.DenseMatrix
import scala.util.Random
import estuary.implicits._

/**
  * Created by mengpan on 2017/10/27.
  */
trait Reader {
  def read(filePath: String): (DenseMatrix[Double], DenseMatrix[Double])

  protected def getMatchedFilesName(filePath: String): Seq[String] = {
    val endDirIndex = filePath.lastIndexOf('/')
    val dirPath = filePath.substring(0, endDirIndex + 1)
    val fileRegex = filePath.r
    val allFiles = new File(dirPath).listFiles().map(_.getCanonicalPath)
    (for (path <- allFiles) yield {
      fileRegex.findFirstIn(path)
    }).filter {
      case Some(s) => true
      case None => false
    }.map(_.get)
  }

  def partition(srcFile: String, n: Int): Unit = {
    val endDirIndex = srcFile.lastIndexOf('/')
    val dirPath = srcFile.substring(0, endDirIndex + 1)
    val (feature, label) = read(srcFile)
    val numExamples = feature.rows
    val shuffledIndex = Random.shuffle[Int, Vector]((0 until numExamples).toVector)
    val nn = numExamples

    for (i <- 0 until n) yield {
      feature(shuffledIndex.slice(i * nn / n, (i + 1) * nn / n), ::).toDenseMatrix.save(s"${dirPath}/${i + 1}-feature.dat")
      label(shuffledIndex.slice(i * nn / n, (i + 1) * nn / n), ::).toDenseMatrix.save(s"${dirPath}/${i + 1}-label.dat")
    }
  }
}
