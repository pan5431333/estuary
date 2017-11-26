package estuary.data

/**
  * Created by mengpan on 2017/10/29.
  */
class MacCifarTenReader extends CifarTenReader{
  override protected val trainingLabelPath: String = """/Users/mengpan/Downloads/trainLabels.csv"""
  override protected val labelFilteringSeq: Seq[Int] = Seq(1, 2)
}
