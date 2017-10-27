package estuary.data

import java.io.File

import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/10/27.
  */
trait Reader {
  def read(filePath: String): (DenseMatrix[Double], DenseMatrix[Double])

  protected def getMatchedFilesName(filePath: String): Seq[String] = {
    val endDirIndex = filePath.lastIndexOf('/')
    val dirPath = filePath.substring(0, endDirIndex+1)
    val fileRegex = filePath.r
    val allFiles = new File(dirPath).listFiles().map(_.getCanonicalPath)
    (for (path <- allFiles) yield {
      fileRegex.findFirstIn(path)
    }).filter{
      case Some(s) => true
      case None => false
    }.map(_.get)
  }
}
