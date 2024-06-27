using System;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Linq;


class Program
{
    static void Main()
    {
        var sourceName = "sample-1.jpg";
        using (var source = Cv2.ImRead(Path.Combine("images", sourceName), ImreadModes.Color))
        {
            foreach (var i in Enumerable.Range(2, 10 - 1))
            {
                var targetName = $"sample-{i}.jpg";
                using (var target = Cv2.ImRead(Path.Combine("images", targetName), ImreadModes.Color))
                {
                    Console.WriteLine($"[Compare `{sourceName}` vs `{targetName}`]");

                    // calculate SSIM
                    using (var gray1 = new Mat())
                    using (var gray2 = new Mat())
                    {
                        Cv2.CvtColor(source, gray1, ColorConversionCodes.BGR2GRAY);
                        Cv2.CvtColor(target, gray2, ColorConversionCodes.BGR2GRAY);

                        var ssimScore = CalculateSSIM(gray1, gray2);
                        Console.WriteLine($"SSIM: {ssimScore}\n");
                    }
                }
            }

            foreach (var i in Enumerable.Range(1, 5))
            {
                var targetName = $"move-{i}.jpg";
                using (var target = Cv2.ImRead(Path.Combine("images", targetName), ImreadModes.Color))
                {
                    Console.WriteLine($"[Compare `{sourceName}` vs `{targetName}`]");

                    // calculate SSIM without align
                    using (var gray1 = new Mat())
                    using (var gray2 = new Mat())
                    {
                        Cv2.CvtColor(source, gray1, ColorConversionCodes.BGR2GRAY);
                        Cv2.CvtColor(target, gray2, ColorConversionCodes.BGR2GRAY);

                        var ssimScore = CalculateSSIM(gray1 , gray2);
                        Console.WriteLine($"SSIM: {ssimScore}");
                    }

                    // calculate SSIM with align
                    using (var gray1 = new Mat())
                    using (var gray2 = new Mat())
                    {
                        var alignedTarget = AlignImages(source, target);

                        Cv2.CvtColor(source, gray1, ColorConversionCodes.BGR2GRAY);
                        Cv2.CvtColor(alignedTarget, gray2, ColorConversionCodes.BGR2GRAY);

                        var ssimScore = CalculateSSIM(gray1, gray2);
                        Console.WriteLine($"SSIM(align): {ssimScore}\n");
                    }
                }
            }
        }
    }

    static Mat AlignImages(Mat source, Mat target)
    {
        // Convert images to grayscale
        Mat gray1 = new Mat();
        Mat gray2 = new Mat();
        Cv2.CvtColor(source, gray1, ColorConversionCodes.BGR2GRAY);
        Cv2.CvtColor(target, gray2, ColorConversionCodes.BGR2GRAY);

        // Initialize the warp matrix
        Mat warpMatrix = Mat.Eye(2, 3, MatType.CV_32F);

        // Set the number of iterations and termination criteria
        var criteria = new TermCriteria(
            Type: CriteriaTypes.Eps | CriteriaTypes.Count,
            MaxCount: 5000,
            Epsilon: 1e-10);

        // Run the ECC algorithm to find the transformation matrix
        Cv2.FindTransformECC(
            templateImage: gray1,
            inputImage: gray2,
            warpMatrix: warpMatrix,
            motionType: MotionTypes.Translation,
            criteria: criteria);

        // Apply the warp transformation
        Mat alignedImage = new Mat();
        Cv2.WarpAffine(
            src: target, 
            dst: alignedImage,
            m: warpMatrix,
            dsize: source.Size(),
            flags: InterpolationFlags.Linear | InterpolationFlags.WarpInverseMap);

        return alignedImage;
    }

    static double CalculateSSIM(Mat img1, Mat img2)
    {
        double C1 = 6.5025, C2 = 58.5225;
        Mat I1 = new Mat();
        Mat I2 = new Mat();
        img1.ConvertTo(I1, MatType.CV_32F);
        img2.ConvertTo(I2, MatType.CV_32F);

        Mat I1_2 = I1.Mul(I1);
        Mat I2_2 = I2.Mul(I2);
        Mat I1_I2 = I1.Mul(I2);

        Mat mu1 = new Mat();
        Mat mu2 = new Mat();
        Cv2.GaussianBlur(I1, mu1, new Size(11, 11), 1.5);
        Cv2.GaussianBlur(I2, mu2, new Size(11, 11), 1.5);

        Mat mu1_2 = mu1.Mul(mu1);
        Mat mu2_2 = mu2.Mul(mu2);
        Mat mu1_mu2 = mu1.Mul(mu2);

        Mat sigma1_2 = new Mat();
        Mat sigma2_2 = new Mat();
        Mat sigma12 = new Mat();

        Cv2.GaussianBlur(I1_2, sigma1_2, new Size(11, 11), 1.5);
        Cv2.GaussianBlur(I2_2, sigma2_2, new Size(11, 11), 1.5);
        Cv2.GaussianBlur(I1_I2, sigma12, new Size(11, 11), 1.5);

        sigma1_2 -= mu1_2;
        sigma2_2 -= mu2_2;
        sigma12 -= mu1_mu2;

        Mat t1 = 2 * mu1_mu2 + Scalar.All(C1);
        Mat t2 = 2 * sigma12 + Scalar.All(C2);
        Mat t3 = t1.Mul(t2);

        t1 = mu1_2 + mu2_2 + Scalar.All(C1);
        t2 = sigma1_2 + sigma2_2 + Scalar.All(C2);
        t1 = t1.Mul(t2);

        Mat ssim_map = new Mat();
        Cv2.Divide(t3, t1, ssim_map);
        Scalar mssim = Cv2.Mean(ssim_map);

        return mssim.Val0;
    }
}