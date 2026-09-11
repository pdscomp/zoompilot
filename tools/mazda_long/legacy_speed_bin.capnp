# zoompilot: the LateralTorqueParameters layout logs carried before 2026-09, when the
# speed-bin fields lived on comma's struct as ordinals @14 to @18. Those legacy fields remain
# declared for old-log readers, while new producers publish bins on customReserved19 and leave
# the inline fields empty. This copy preserves the old layout for offline tools
# (speed_bin_log.py). The struct id is a fresh one on purpose: this file is never loaded
# beside log.capnp.
@0xd8c4b0b6a3c58e21;

struct LegacyLateralTorqueParameters {
  valid @0 :Bool;
  latAccelFactorRaw @1 :Float32;
  latAccelOffsetRaw @2 :Float32;
  frictionCoefficientRaw @3 :Float32;
  latAccelFactorFiltered @4 :Float32;
  latAccelOffsetFiltered @5 :Float32;
  frictionCoefficientFiltered @6 :Float32;
  totalBucketPoints @7 :Float32;
  decay @8 :Float32;
  maxResets @9 :Float32;
  points @10 :List(List(Float32));
  version @11 :Int32;
  useParams @12 :Bool;
  calPerc @13 :Int8;
  speedBinCenters @14 :List(Float32);
  speedBinLatAccelFactors @15 :List(Float32);
  speedBinFrictions @16 :List(Float32);
  speedBinValid @17 :List(Bool);
  speedBinPoints @18 :List(List(List(Float32)));
}
