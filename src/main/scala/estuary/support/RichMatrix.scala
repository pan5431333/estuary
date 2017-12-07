package estuary.support

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix

import scala.io.Source

/**
  * Created by mengpan on 2017/10/27.
  */
class RichMatrix(data: Array[Double], nRows: Int, nCols: Int) {
  def save(path: String): Unit = {
    val writer = new PrintWriter(new File(path))
    writer.println("nRows: " + nRows)
    writer.println("nCols: " + nCols)
    writer.print("data: ")
    data.foreach{ d =>
      writer.print(d)
      writer.print(',')
    }
    writer.close()
  }

  def toDenseMatrix: DenseMatrix[Double] = {
    new DenseMatrix[Double](nRows, nCols, data)
  }
}

object RichMatrix {
  def apply(data: Array[Double], nRows: Int, nCols: Int): RichMatrix = {
    new RichMatrix(data, nRows, nCols)
  }

  def read(path: String): RichMatrix = {
    var nRows: Int = 0
    var nCols: Int = 0
    var data: Array[Double] = new Array[Double](1)
    val lines = Source.fromFile(path).getLines()
    for (line <- lines) {
      val indicatorIndex = line.indexOf(':')
      val indicator = line.substring(0, indicatorIndex)
      indicator match {
        case "nRows" => nRows = line.substring(indicatorIndex+2).trim().toInt
        case "nCols" => nCols = line.substring(indicatorIndex+2).trim().toInt
        case "data" => data = line.substring(indicatorIndex+2).split(',').map(_.trim().toDouble)
      }
    }
    new RichMatrix(data, nRows, nCols)
  }
}
