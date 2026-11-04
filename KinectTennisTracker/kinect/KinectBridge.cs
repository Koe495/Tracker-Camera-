using System;
using System.IO;
using System.Threading;
using Microsoft.Kinect;

class KinectBridge
{
    private const int Width = 640;
    private const int Height = 480;
    private const int Pixels = Width * Height;
    private const ColorImageFormat ColorFormat = ColorImageFormat.RgbResolution640x480Fps30;
    private const DepthImageFormat DepthFormat = DepthImageFormat.Resolution640x480Fps30;

    static int Main(string[] args)
    {
        try
        {
            var sensors = KinectSensor.KinectSensors;
            KinectSensor connected = null;
            foreach (var sensor in sensors)
            {
                Console.Error.WriteLine("Kinect status: " + sensor.Status);
                if (connected == null && sensor.Status == KinectStatus.Connected)
                    connected = sensor;
            }
            if (connected == null)
            {
                Console.Error.WriteLine("SDK v1.8 found no connected Kinect sensor.");
                return 2;
            }
            if (args.Length == 1 && args[0] == "probe") return 0;
            bool project = args.Length == 1 && args[0] == "live-project";
            bool live = project || (args.Length == 1 && args[0] == "live");
            int count = int.MaxValue;
            if (!live && (args.Length != 2 || args[0] != "stream" ||
                          !int.TryParse(args[1], out count) || count < 1))
                throw new ArgumentException("Usage: KinectBridge.exe probe | live | live-project | stream <frame-count>");
            Capture(connected, count, live, project);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }

    static unsafe void Capture(KinectSensor sensor, int count, bool live, bool project)
    {
        sensor.ColorStream.Enable(ColorFormat);
        sensor.DepthStream.Enable(DepthFormat);
        var color = new byte[Pixels * 4];
        var depth = new DepthImagePixel[Pixels];
        var depthMillimeters = new byte[Pixels * 2];
        var aligned = new SkeletonPoint[Pixels];
        var xyz = new byte[Pixels * 12];
        var gate = new object();
        int captured = 0;
        long lastColor = -1;
        long lastDepth = -1;
        Exception failure = null;
        bool closing = false;
        using (var progress = new AutoResetEvent(false))
        using (var stop = new ManualResetEvent(false))
        using (var output = new BinaryWriter(Console.OpenStandardOutput()))
        {
            Thread stopReader = null;
            if (live)
            {
                stopReader = new Thread(() =>
                {
                    try { Console.ReadLine(); }
                    catch (IOException) { }
                    lock (gate)
                        if (!closing) stop.Set();
                });
                stopReader.IsBackground = true;
                stopReader.Start();
            }
            var signals = new WaitHandle[] { progress, stop };
            EventHandler<AllFramesReadyEventArgs> handle = (sender, frames) =>
            {
                lock (gate)
                {
                    if (closing || failure != null || captured >= count) return;
                    try
                    {
                        using (var c = frames.OpenColorImageFrame())
                        using (var d = frames.OpenDepthImageFrame())
                        {
                            if (c == null || d == null || c.Timestamp <= lastColor || d.Timestamp <= lastDepth ||
                                Math.Abs(c.Timestamp - d.Timestamp) > 25) return;
                            c.CopyPixelDataTo(color);
                            d.CopyDepthImagePixelDataTo(depth);
                            var mapper = sensor.CoordinateMapper;
                            mapper.MapColorFrameToSkeletonFrame(ColorFormat, DepthFormat, depth, aligned);
                            if (project && captured == 0) WriteProjectionSamples(output, mapper);
                            fixed (byte* depthBytes = depthMillimeters, xyzBytes = xyz)
                            {
                                ushort* mm = (ushort*)depthBytes;
                                float* points = (float*)xyzBytes;
                                for (int i = 0; i < Pixels; i++)
                                {
                                    int z = depth[i].Depth;
                                    mm[i] = (ushort)(z >= d.MinDepth && z <= d.MaxDepth ? z : 0);
                                    // SDK +X points to camera-left. Export image-right, up, forward.
                                    points[3 * i] = -aligned[i].X;
                                    points[3 * i + 1] = aligned[i].Y;
                                    points[3 * i + 2] = aligned[i].Z;
                                }
                            }
                            output.Write(new byte[] { 82, 71, 66, 68 }); // RGBD
                            for (int i = 0; i < 2; i++)
                            {
                                output.Write((ushort)Width);
                                output.Write((ushort)Height);
                            }
                            output.Write(c.Timestamp);
                            output.Write(d.Timestamp);
                            output.Write(color);
                            output.Write(depthMillimeters);
                            output.Write(xyz);
                            output.Flush();
                            lastColor = c.Timestamp;
                            lastDepth = d.Timestamp;
                            captured++;
                        }
                    }
                    catch (Exception error) { failure = error; }
                    progress.Set();
                }
            };
            sensor.AllFramesReady += handle;
            try
            {
                sensor.Start();
                while (true)
                {
                    int signal = WaitHandle.WaitAny(signals, TimeSpan.FromSeconds(8));
                    if (signal == 1) break;
                    if (signal == WaitHandle.WaitTimeout)
                        throw new TimeoutException("No accepted RGB-D pair for 8 seconds. Check USB/power or close other Kinect apps.");
                    lock (gate)
                    {
                        if (failure != null) throw failure;
                        if (captured >= count) break;
                    }
                }
            }
            finally
            {
                lock (gate) closing = true;
                sensor.AllFramesReady -= handle;
                if (sensor.IsRunning) sensor.Stop();
                if (stopReader != null && stop.WaitOne(0)) stopReader.Join();
            }
        }
    }

    static void WriteProjectionSamples(BinaryWriter output, CoordinateMapper mapper)
    {
        float[] levels = { -0.35f, -0.175f, 0f, 0.175f, 0.35f };
        float[] depths = { 0.9f, 1.4f, 2.1f, 3.0f, 3.8f };
        output.Write(new byte[] { 80, 82, 74, 49 }); // PRJ1
        output.Write(levels.Length * levels.Length * depths.Length);
        foreach (float z in depths)
        foreach (float y in levels)
        foreach (float x in levels)
        {
            var p = new SkeletonPoint { X = -x * z, Y = y * z, Z = z };
            var uv = mapper.MapSkeletonPointToColorPoint(p, ColorFormat);
            output.Write(-p.X);
            output.Write(p.Y);
            output.Write(p.Z);
            output.Write(uv.X);
            output.Write(uv.Y);
        }
    }
}
