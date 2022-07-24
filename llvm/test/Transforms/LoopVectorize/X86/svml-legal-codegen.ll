; Check that vector codegen splits illegal sin8 call to two sin4 calls on AVX for double datatype.
; The C code used to generate this test:

; #include <math.h>
;
; void foo(double *a, int N){
;   int i;
; #pragma clang loop vectorize_width(8)
;   for (i=0;i<N;i++){
;     a[i] = sin(i);
;   }
; }

; RUN: opt -vector-library=SVML -inject-tli-mappings -loop-vectorize -force-vector-width=8 -mattr=avx -S < %s | FileCheck %s

; CHECK: [[I1:%.*]] = sitofp <8 x i32> [[I0:%.*]] to <8 x double>
; CHECK-NEXT: [[S1:%shuffle.*]] = shufflevector <8 x double> [[I1]], <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: [[I2:%.*]] = call fast intel_svmlcc256 <4 x double> @__svml_sin4(<4 x double> [[S1]])
; CHECK-NEXT: [[S2:%shuffle.*]] = shufflevector <8 x double> [[I1]], <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT: [[I3:%.*]] = call fast intel_svmlcc256 <4 x double> @__svml_sin4(<4 x double> [[S2]])
; CHECK-NEXT: [[comb:%combined.*]] = shufflevector <4 x double> [[I2]], <4 x double> [[I3]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: store <8 x double> [[comb]], <8 x double>* [[TMP:%.*]], align 8


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @foo(double* nocapture %a, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp5 = icmp sgt i32 %N, 0
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %call = tail call fast double @sin(double %conv) #2
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %call, double* %arrayidx, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !6

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Function Attrs: nounwind
declare dso_local double @sin(double) local_unnamed_addr #1

!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.vectorize.width", i32 8}
