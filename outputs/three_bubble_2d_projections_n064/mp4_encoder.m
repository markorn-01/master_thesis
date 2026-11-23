#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>
#import <ImageIO/ImageIO.h>
#import <CoreVideo/CoreVideo.h>
#include <unistd.h>
int main(int argc,const char **argv) {
 @autoreleasepool {
  if(argc!=5) { fprintf(stderr,"frames-dir output.mp4 readback-dir frame-count\n"); return 1; }
  NSString *folder=@(argv[1]), *target=@(argv[2]), *readback=@(argv[3]); int count=atoi(argv[4]);
  NSError *error=nil; [[NSFileManager defaultManager] createDirectoryAtPath:readback withIntermediateDirectories:YES attributes:nil error:&error];
  NSString *first=[folder stringByAppendingPathComponent:@"event_0000.png"];
  CGImageSourceRef source=CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:first],NULL);
  CGImageRef image=CGImageSourceCreateImageAtIndex(source,0,NULL); size_t width=CGImageGetWidth(image),height=CGImageGetHeight(image); CGImageRelease(image);CFRelease(source);
  AVAssetWriter *writer=[[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:target] fileType:AVFileTypeMPEG4 error:&error];
  NSDictionary *settings=@{AVVideoCodecKey:AVVideoCodecTypeH264,AVVideoWidthKey:@(width),AVVideoHeightKey:@(height),AVVideoCompressionPropertiesKey:@{AVVideoAverageBitRateKey:@3000000,AVVideoMaxKeyFrameIntervalKey:@1,AVVideoAllowFrameReorderingKey:@NO}};
  AVAssetWriterInput *input=[AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:settings];input.expectsMediaDataInRealTime=NO;
  AVAssetWriterInputPixelBufferAdaptor *adaptor=[AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input sourcePixelBufferAttributes:@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32ARGB),(id)kCVPixelBufferWidthKey:@(width),(id)kCVPixelBufferHeightKey:@(height),(id)kCVPixelBufferCGImageCompatibilityKey:@YES,(id)kCVPixelBufferCGBitmapContextCompatibilityKey:@YES}];
  if(!writer || ![writer canAddInput:input]) { NSLog(@"Writer setup failed: %@",error); return 2; }
  [writer addInput:input];if(![writer startWriting]) {NSLog(@"%@",writer.error);return 3;}[writer startSessionAtSourceTime:kCMTimeZero];
  for(int i=0;i<count;i++) {
   NSString *path=[folder stringByAppendingPathComponent:[NSString stringWithFormat:@"event_%04d.png",i]];
   source=CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:path],NULL);if(!source)return 4;
   image=CGImageSourceCreateImageAtIndex(source,0,NULL);CVPixelBufferRef buffer=NULL;
   CVPixelBufferCreate(kCFAllocatorDefault,width,height,kCVPixelFormatType_32ARGB,(__bridge CFDictionaryRef)@{(id)kCVPixelBufferCGImageCompatibilityKey:@YES,(id)kCVPixelBufferCGBitmapContextCompatibilityKey:@YES},&buffer);
   CVPixelBufferLockBaseAddress(buffer,0);CGColorSpaceRef space=CGColorSpaceCreateDeviceRGB();
   CGContextRef context=CGBitmapContextCreate(CVPixelBufferGetBaseAddress(buffer),width,height,8,CVPixelBufferGetBytesPerRow(buffer),space,kCGImageAlphaNoneSkipFirst|kCGBitmapByteOrder32Big);
   CGContextDrawImage(context,CGRectMake(0,0,width,height),image);CGContextRelease(context);CGColorSpaceRelease(space);CVPixelBufferUnlockBaseAddress(buffer,0);
   CFAbsoluteTime deadline=CFAbsoluteTimeGetCurrent()+30;
   while(!input.readyForMoreMediaData && writer.status==AVAssetWriterStatusWriting && CFAbsoluteTimeGetCurrent()<deadline)usleep(10000);
   if(![adaptor appendPixelBuffer:buffer withPresentationTime:CMTimeMake(i,1)]) {NSLog(@"Append failed: %@",writer.error);return 5;}
   CVPixelBufferRelease(buffer);CGImageRelease(image);CFRelease(source);
  }
  [writer endSessionAtSourceTime:CMTimeMake(count,1)];[input markAsFinished];dispatch_semaphore_t done=dispatch_semaphore_create(0);
  [writer finishWritingWithCompletionHandler:^{dispatch_semaphore_signal(done);}];dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC));
  if(writer.status!=AVAssetWriterStatusCompleted){NSLog(@"Finish failed: %@",writer.error);return 6;}
  AVURLAsset *asset=[AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:target] options:nil];AVAssetTrack *track=[[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
  AVAssetReader *reader=[[AVAssetReader alloc] initWithAsset:asset error:&error];AVAssetReaderTrackOutput *output=[AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:track outputSettings:@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA)}];[reader addOutput:output];[reader startReading];
  NSMutableArray *times=[NSMutableArray array];CMSampleBufferRef sample=NULL;int decoded=0;
  while((sample=[output copyNextSampleBuffer])) {
   [times addObject:@(CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample)))];
   CVPixelBufferRef pix=CMSampleBufferGetImageBuffer(sample);CVPixelBufferLockBaseAddress(pix,kCVPixelBufferLock_ReadOnly);CGColorSpaceRef space=CGColorSpaceCreateDeviceRGB();
   CGContextRef context=CGBitmapContextCreate(CVPixelBufferGetBaseAddress(pix),width,height,8,CVPixelBufferGetBytesPerRow(pix),space,kCGBitmapByteOrder32Little|kCGImageAlphaPremultipliedFirst);CGImageRef cg=CGBitmapContextCreateImage(context);
   NSString *path=[readback stringByAppendingPathComponent:[NSString stringWithFormat:@"decoded_%04d.png",decoded]];
   CGImageDestinationRef dest=CGImageDestinationCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:path],CFSTR("public.png"),1,NULL);CGImageDestinationAddImage(dest,cg,NULL);CGImageDestinationFinalize(dest);CFRelease(dest);CGImageRelease(cg);CGContextRelease(context);CGColorSpaceRelease(space);CVPixelBufferUnlockBaseAddress(pix,kCVPixelBufferLock_ReadOnly);CFRelease(sample);decoded++;
  }
  NSDictionary *report=@{@"codec":@"H.264",@"encoder":@"macOS AVFoundation",@"decoded_frames":@(decoded),@"width":@(width),@"height":@(height),@"presentation_times_seconds":times,@"duration_seconds":@(CMTimeGetSeconds(asset.duration)),@"reader_completed":@(reader.status==AVAssetReaderStatusCompleted)};
  NSData *data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];[data writeToFile:[readback stringByAppendingPathComponent:@"encoding.json"] atomically:YES];
  printf("Encoded and decoded %d frames.\n",decoded);return decoded==count?0:7;
 }
}
