package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.min
import kotlin.math.floor

import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

data class Detection(val box: RectF, val score: Float, val classId: Int)

class YoloDetector(
    context: Context,
    modelName: String = "yolo26n.tflite",
    private val inputSize: Int = 640, // ----------
    private val threshold: Float = 0.35f
) {
    private val interpreter: Interpreter

    private var gpuDelegate: GpuDelegate? = null

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())

    private val lut = FloatArray(256) { it / 255f }
    private val chw = FloatArray(3 * inputSize * inputSize)

    private val floatBuf = inputBuffer.asFloatBuffer()

    private val srcMap = IntArray(inputSize * inputSize)
    private var mapW = -1
    private var mapH = -1
    private var mapStride = -1
    private var mapRot = -1

    private var uw = 0
    private var uh = 0
    private var scale = 1f
    private var padX = 0f
    private var padY = 0f

    private val output = Array(1){Array(300){ FloatArray(6) } }

    private fun rebuildMap(w: Int, h: Int, stride: Int, rotation: Int){
        val swap = rotation % 100 != 0
        uw = if (swap) h else w
        uh = if (swap) w else h
        scale = min(inputSize / uw.toFloat(), inputSize / uh.toFloat())
        padX = (inputSize - uw * scale) / 2f
        padY = (inputSize - uh * scale) / 2f

        for (oy in 0 until inputSize) {
            val uy = floor((oy + 0.5f - padY) / scale).toInt()
            for (ox in 0 until inputSize) {
                val ux = floor((ox + 0.5f - padX) / scale).toInt()
                val idx = oy * inputSize + ox
                if (ux !in 0 until uw || uy !in 0 until uh){
                    srcMap[idx] = -1
                    continue
                }

                val sx: Int
                val sy: Int
                when (rotation){
                    90 -> { sx = uy; sy = h - 1 - ux}
                    180 -> {sx = w - 1 - ux; sy = h - 1 - uy}
                    270 -> {sx = w - 1 - uy; sy = ux}
                    else -> {sx = ux; sy = uy}

                }
                srcMap[idx] = sy * stride + sx * 4
            }
        }
        mapW = w
        mapH = h
        mapStride = stride
        mapRot = rotation
    }

    init {
        val options = Interpreter.Options()
        var usingGpu = false

        CompatibilityList().use {compat ->
            if (compat.isDelegateSupportedOnThisDevice){
                try {
                    val gpuOption = compat.bestOptionsForThisDevice.apply {
                        setSerializationParams(
                            context.cacheDir.absolutePath,
                            "yolo26n_640_v1"
                        )
                    }
                    gpuDelegate = GpuDelegate(gpuOption)
                    options.addDelegate(gpuDelegate)
                    usingGpu = true
                } catch (e: Exception){
                    android.util.Log.e("YOLO", "GPU не запустился", e)
                    gpuDelegate?.close()
                    gpuDelegate = null
                }
            }

        }

        if (!usingGpu) options.setNumThreads(4)
        android.util.Log.d("YOLO", "GPU delegate: $usingGpu")

        interpreter = Interpreter(loadModel(context, modelName), options)

        val shape = interpreter.getOutputTensor(0).shape()
        val inshape = interpreter.getInputTensor(0).shape()

        require(shape.size == 3 && shape[1] == 300 && shape[2] == 6){
            "Неожиданная форма выхода"
        }
        require(inshape.contentEquals(intArrayOf(1, 3, inputSize, inputSize))){
            "Неожиданный размер изображения"
        }
    }
    private fun loadModel(context: Context, name: String): ByteBuffer {
        val fd = context.assets.openFd(name)
        FileInputStream(fd.fileDescriptor).use {stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
    }

    fun detect(rgba: ByteArray, w: Int, h: Int, rowStride: Int, rotation: Int): List<Detection>{
        if (w != mapW || h!= mapH || rowStride != mapStride || rotation != mapRot){
            rebuildMap(w, h, rowStride, rotation)
        }

// в detect(), вместо цикла с floatBuf.put(i, ...):
        val n = inputSize * inputSize
        val padVal = lut[114]
        for (i in 0 until n) {
            val o = srcMap[i]
            if (o < 0){
                chw[i] = padVal; chw[n + i] = padVal; chw[2 * n + i] = padVal
            }else {

                chw[i] = lut[rgba[o].toInt() and 0xFF]
                chw[n + i] = lut[rgba[o + 1].toInt() and 0xFF]
                chw[2 * n + i] = lut[rgba[o + 2].toInt() and 0xFF]
            }
        }
        floatBuf.rewind()
        floatBuf.put(chw)
        inputBuffer.rewind()

        interpreter.run(inputBuffer, output)

        val result = ArrayList<Detection>()
        for (row in output[0]){
            if (row[5].toInt() != 0) continue

            val score = row[4]
            if (score < threshold) continue

            val x1 = row[0]; val y1 = row[1]; val x2 = row[2]; val y2 = row[3]

            val box = RectF(
                ((x1 - padX) / scale).coerceIn(0f, uw.toFloat()),
                ((y1 - padY) / scale).coerceIn(0f, uh.toFloat()),
                ((x2 - padX) / scale).coerceIn(0f, uw.toFloat()),
                ((y2 - padY) / scale).coerceIn(0f, uh.toFloat())
            )

            result.add(Detection(box, score, row[5].toInt()))
        }

        return result
    }
    fun close(){
        interpreter.close()
        gpuDelegate?.close()
    }
}

val COCO_LABELS = listOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
)