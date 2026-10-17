package sim

import (
	"testing"
)

func TestDetectionQuick(t *testing.T) {
	cfg := Default()
	rep := Detection(cfg, Healthy(1000, 12, 3000), Healthy(2000, 8, 3000), Faulty(cfg, 3000, 3, 3000))
	for i, n := range rep.Names {
		caught, wrong, unl, fa := 0, 0, 0, 0
		for _, v := range rep.Verdicts[i] {
			if v.Caught {
				caught++
			}
			if v.WrongFirst {
				wrong++
			}
			if v.Unlocated {
				unl++
			}
			fa += v.FalseBefore
		}
		t.Logf("%-36s h=%v caught %d wrong %d unloc %d of %d, falseBefore %d healthyFA %d", n, rep.Thresholds[i], caught, wrong, unl, len(rep.Verdicts[i]), fa, rep.HealthyAlarms[i])
	}
	for _, v := range rep.Verdicts[0] {
		t.Logf("%v %.2f start %d end %d caught %v wrong %v delay %d kind %v", v.Fault.Kind, v.Fault.Factor, v.Fault.Start, v.Fault.End, v.Caught, v.WrongFirst, v.Delay, v.KindRight)
	}
}
