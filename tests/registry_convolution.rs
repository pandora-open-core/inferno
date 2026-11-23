//! Independent grouped/dilated convolution parity for the registry backend.
use inferno::candle::{Device, Tensor};
use inferno::convolution::{conv1d_diff, conv2d_diff};

#[test]
fn grouped_dilated_convolution_matches_scalar_reference() -> inferno::candle::Result<()> {
    let (batch, channels, outputs, length, taps, groups) = (2, 4, 6, 7, 3, 2);
    let (padding, stride, dilation) = (2, 2, 2);
    let data: Vec<f64> = (0..batch * channels * length)
        .map(|i| i as f64 / 31. - 0.7)
        .collect();
    let weights: Vec<f64> = (0..outputs * (channels / groups) * taps)
        .map(|i| (i as f64 * 0.3).sin())
        .collect();
    let x = Tensor::from_vec(data.clone(), (batch, channels, length), &Device::Cpu)?;
    let w = Tensor::from_vec(
        weights.clone(),
        (outputs, channels / groups, taps),
        &Device::Cpu,
    )?;
    let actual = conv1d_diff(&x, &w, padding, stride, dilation, groups)?;
    let out_len = (length + 2 * padding - dilation * (taps - 1) - 1) / stride + 1;
    assert_eq!(actual.dims(), &[batch, outputs, out_len]);
    let mut expected = Vec::new();
    for b in 0..batch {
        for co in 0..outputs {
            let group = co / (outputs / groups);
            for pos in 0..out_len {
                let mut sum = 0.;
                for ci in 0..channels / groups {
                    for tap in 0..taps {
                        let offset = pos * stride + tap * dilation;
                        if offset >= padding && offset - padding < length {
                            let index = (b * channels + group * (channels / groups) + ci) * length
                                + offset
                                - padding;
                            sum +=
                                data[index] * weights[(co * (channels / groups) + ci) * taps + tap];
                        }
                    }
                }
                expected.push(sum);
            }
        }
    }
    for (a, e) in actual.flatten_all()?.to_vec1::<f64>()?.iter().zip(expected) {
        assert!((a - e).abs() < 1e-12, "{a} != {e}");
    }
    Ok(())
}

#[test]
fn grouped_dilated_conv2d_matches_scalar_reference_and_gradients() -> inferno::candle::Result<()> {
    use inferno::candle::Var;
    let (batch, channels, outputs, height, width, kh, kw, groups) = (2, 4, 6, 5, 7, 2, 3, 2);
    let (padding, stride, dilation) = (1, 2, 2);
    let data: Vec<f64> = (0..batch * channels * height * width)
        .map(|i| (i as f64 * 0.11).cos())
        .collect();
    let weights: Vec<f64> = (0..outputs * (channels / groups) * kh * kw)
        .map(|i| (i as f64 * 0.3).sin())
        .collect();
    let input = |v: Vec<f64>| Tensor::from_vec(v, (batch, channels, height, width), &Device::Cpu);
    let kernel =
        |v: Vec<f64>| Tensor::from_vec(v, (outputs, channels / groups, kh, kw), &Device::Cpu);
    let x = Var::from_tensor(&input(data.clone())?)?;
    let w = Var::from_tensor(&kernel(weights.clone())?)?;
    let actual = conv2d_diff(&x, &w, padding, stride, dilation, groups)?;
    let oh = (height + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    let ow = (width + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
    assert_eq!(actual.dims(), &[batch, outputs, oh, ow]);
    let mut expected = Vec::new();
    for b in 0..batch {
        for co in 0..outputs {
            let group = co / (outputs / groups);
            for row in 0..oh {
                for col in 0..ow {
                    let mut sum = 0.;
                    for ci in 0..channels / groups {
                        for kr in 0..kh {
                            for kc in 0..kw {
                                let r = row * stride + kr * dilation;
                                let c = col * stride + kc * dilation;
                                if r >= padding
                                    && r - padding < height
                                    && c >= padding
                                    && c - padding < width
                                {
                                    let index = ((b * channels + group * (channels / groups) + ci)
                                        * height
                                        + r
                                        - padding)
                                        * width
                                        + c
                                        - padding;
                                    let wi = ((co * (channels / groups) + ci) * kh + kr) * kw + kc;
                                    sum += data[index] * weights[wi];
                                }
                            }
                        }
                    }
                    expected.push(sum);
                }
            }
        }
    }
    for (a, e) in actual.flatten_all()?.to_vec1::<f64>()?.iter().zip(expected) {
        assert!((a - e).abs() < 1e-12, "{a} != {e}");
    }
    let gradients = actual.sqr()?.sum_all()?.backward()?;
    let loss = |x: &Tensor, w: &Tensor| -> inferno::candle::Result<f64> {
        conv2d_diff(x, w, padding, stride, dilation, groups)?
            .sqr()?
            .sum_all()?
            .to_scalar()
    };
    for (variable, values, is_input) in [(&x, &data, true), (&w, &weights, false)] {
        let gradient = gradients
            .get(variable.as_tensor())
            .unwrap()
            .flatten_all()?
            .to_vec1::<f64>()?;
        let coordinates = if is_input {
            [
                width + 1,
                values.len() / 2 + width + 1,
                values.len() - height * width + (height - 2) * width + width - 2,
            ]
        } else {
            [0, values.len() / 2, values.len() - 1]
        };
        assert!(coordinates.iter().any(|&i| gradient[i].abs() > 1e-8));
        for i in coordinates {
            let mut plus = values.clone();
            let mut minus = values.clone();
            let step = 1e-4;
            plus[i] += step;
            minus[i] -= step;
            let fd = if is_input {
                (loss(&input(plus)?, &w)? - loss(&input(minus)?, &w)?) / (2. * step)
            } else {
                (loss(&x, &kernel(plus)?)? - loss(&x, &kernel(minus)?)?) / (2. * step)
            };
            assert!(
                (gradient[i] - fd).abs() < 1e-6 * fd.abs().max(1.),
                "gradient {} != {fd}",
                gradient[i]
            );
        }
    }
    assert!(conv2d_diff(&x, &w, padding, 0, dilation, groups).is_err());
    assert!(conv2d_diff(&x, &w, padding, stride, dilation, 3).is_err());
    Ok(())
}
