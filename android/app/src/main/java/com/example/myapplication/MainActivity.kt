package com.example.myapplication

import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var detector: YoloDetector
    private lateinit var fpsText: TextView
    private lateinit var overlay: OverlayView
    private lateinit var previewView: PreviewView

    private var frameCount = 0
    private var lastFpsTime = SystemClock.elapsedRealtime()

    @Volatile private var toBitmapMs = 0L
    @Volatile private var allDetectMs = 0L

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val processingScope =
        CoroutineScope(SupervisorJob() + inferenceExecutor.asCoroutineDispatcher())

    private val busy = AtomicBoolean(false)

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Нужно разрешение на камеру", Toast.LENGTH_SHORT).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        fpsText = findViewById(R.id.fpsText)
        overlay = findViewById(R.id.overlayView)
        previewView = findViewById(R.id.previewView)

        fpsText.text = "Загрузка модели..."
        processingScope.launch {
            detector = YoloDetector(this@MainActivity)
            withContext(Dispatchers.Main) {
                permissionLauncher.launch(android.Manifest.permission.CAMERA)
            }
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val cameraProvider = providerFuture.get()

            // один селектор для Preview и Analysis, чтобы совпадали пропорции
            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                if (!busy.compareAndSet(false, true)) {
                    imageProxy.close()
                    return@setAnalyzer
                }

                val a0 = SystemClock.elapsedRealtime()
                val bitmap: Bitmap = imageProxy.toBitmap()
                toBitmapMs = SystemClock.elapsedRealtime() - a0
                val rotation = imageProxy.imageInfo.rotationDegrees
                imageProxy.close()

                processingScope.launch {
                    try {
                        val t0 = SystemClock.elapsedRealtime()
                        val dets = detector.detect(bitmap, rotation)
                        allDetectMs = SystemClock.elapsedRealtime() - t0

                        val swap = rotation % 180 != 0
                        val uw = if (swap) bitmap.height else bitmap.width
                        val uh = if (swap) bitmap.width else bitmap.height

                        withContext(Dispatchers.Main) {
                            overlay.update(dets, uw, uh)
                            frameCount++
                            val now = SystemClock.elapsedRealtime()
                            if (now - lastFpsTime >= 1000) {
                                fpsText.text = "FPS: $frameCount | toBitmap: $toBitmapMs | detect: $allDetectMs"
                                frameCount = 0
                                lastFpsTime = now
                            }
                        }
                    } finally {
                        busy.set(false)
                    }
                }
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        processingScope.cancel()
        analysisExecutor.shutdown()
        inferenceExecutor.execute {
            if (::detector.isInitialized) detector.close()
        }
        inferenceExecutor.shutdown()
    }
}