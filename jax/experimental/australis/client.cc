#include "jax/experimental/australis/client.h"

#include <functional>
#include <utility>

#include "jax/experimental/australis/client-private.h"
#include "jax/experimental/australis/internals.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace aux {

absl::Status internal::ToAbslStatus(const xla::Status& s) {
  return s.ok() ? absl::OkStatus()
                : absl::Status(static_cast<absl::StatusCode>(s.code()),
                               s.error_message());
}

using internal::FactoryFn;

class FactoryDefinition {
 public:
  FactoryDefinition(const char* name, FactoryFn factory)
      : name_(name), factory_(factory) {}

  std::string_view name() { return name_; }

  absl::StatusOr<Client> Get() {
    absl::MutexLock l(&mutex_);
    if (!is_initialized_) {
      value_ = factory_();
      is_initialized_ = true;
    }
    if (!value_.ok()) return value_.status();
    return Client(value_->get());
  }

 private:
  const char* name_;
  FactoryFn factory_;
  absl::Mutex mutex_;
  bool is_initialized_ = false;
  absl::StatusOr<std::shared_ptr<xla::PjRtClient>> value_;
};

std::vector<std::unique_ptr<FactoryDefinition>>& get_factories() {
  static auto* result = new std::vector<std::unique_ptr<FactoryDefinition>>;
  return *result;
}

internal::BackendFactoryRegister::BackendFactoryRegister(const char* name,
                                                         FactoryFn factory) {
  get_factories().push_back(std::make_unique<FactoryDefinition>(name, factory));
}

absl::StatusOr<Client> Client::Get(std::string_view name) {
  for (auto& factory : get_factories()) {
    if (name == factory->name()) {
      return factory->Get();
    }
  }
  return absl::UnknownError(absl::StrCat("No such backend: ", name));
}

absl::StatusOr<Client> Client::GetDefault() {
  for (auto& factory : get_factories()) {
    auto client = factory->Get();
    if (client.ok()) {
      return client;
    } else {
      LOG(ERROR) << "Could not init: " << factory->name() << ": "
                 << client.status();
    }
  }
  return absl::UnknownError("No backends available");
}

std::vector<Device> Client::LocalDevices() {
  std::vector<Device> out;
  for (auto* device : client_->addressable_devices()) {
    out.push_back(Device(device));
  }
  return out;
}

absl::StatusOr<std::unique_ptr<Executable>> Client::Compile(
    int num_replicas, int num_partitions, bool use_spmd_partitioning,
    bool tuple_args, std::string_view binary_proto) {
  xla::HloModuleProto proto;
  if (!proto.ParsePartialFromArray(binary_proto.data(), binary_proto.size())) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Could not parse hlo proto.");
  }

  xla::XlaComputation computation(proto);
  xla::CompileOptions compile_options;
  compile_options.parameter_is_tupled_arguments = tuple_args;
  compile_options.executable_build_options.set_num_replicas(num_replicas);
  compile_options.executable_build_options.set_num_partitions(num_partitions);
  compile_options.executable_build_options.set_use_spmd_partitioning(
      use_spmd_partitioning);
  auto result = client_->Compile(computation, compile_options);
  if (!result.ok()) return internal::ToAbslStatus(result.status());
  return std::unique_ptr<Executable>(
      new Executable(std::move(result).ValueOrDie()));
}

std::string_view Client::platform_name() const {
  return client_->platform_name();
}

DeviceArray::DeviceArray(
    absl::InlinedVector<std::unique_ptr<xla::PjRtBuffer>, 1> buffers)
    : buffers_(std::move(buffers)) {}
DeviceArray::DeviceArray() {}
DeviceArray::~DeviceArray() {}
DeviceArray::DeviceArray(DeviceArray&& other)
    : buffers_(std::move(other.buffers_)) {}
DeviceArray& DeviceArray::operator=(DeviceArray&& other) {
  buffers_ = std::move(other.buffers_);
  return *this;
}

absl::StatusOr<DeviceArray> DeviceArray::Create(
    absl::Span<const Array> arrays, absl::Span<const Device> devices) {
  if (arrays.size() != devices.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("buffers.size() != devices.size() --- (", arrays.size(),
                     " != ", devices.size(), ")."));
  }
  if (arrays.empty()) {
    return absl::InvalidArgumentError(
        "Cannot make a DeviceArray with no data!");
  }
  auto* client = devices[0].device()->client();
  for (Device dev : devices) {
    DCHECK(dev.device()->client() == client)
        << "All devices must be from the same client!";
  }
  const ShapedArray& aval = arrays[0].aval();
  for (const Array& array : arrays) {
    if (array.aval().shape != aval.shape || array.aval().dtype != aval.dtype) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot make a DeviceArray with different per-shard dtypes: ",
          aval.ToString(), " vs ", array.aval().ToString()));
    }
  }
  absl::InlinedVector<std::unique_ptr<xla::PjRtBuffer>, 1> shards;
  for (size_t i = 0; i < arrays.size(); ++i) {
    const Array& array = arrays[i];
    auto buf = internal::ToAbslStatusOr(client->BufferFromHostBuffer(
        array.raw_data(), internal::PrimitiveTypeToXla(array.aval().dtype),
        array.aval().shape, array.byte_strides(),
        // Must be strict in order to be safe.
        xla::PjRtClient::HostBufferSemantics::kZeroCopy,
        [owner = array.owner()] {}, devices[i].device()));
    if (!buf.ok()) return buf.status();
    shards.emplace_back(std::move(buf).value());
  }
  return DeviceArray(std::move(shards));
}

absl::StatusOr<DeviceArray> DeviceArray::Create(
    absl::Span<const void* const> buffers, PrimitiveType type,
    absl::Span<int64_t const> dims,
    absl::optional<absl::Span<int64_t const>> byte_strides,
    absl::Span<const Device> devices) {
  if (buffers.size() != devices.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("buffers.size() != devices.size() --- (", buffers.size(),
                     " != ", devices.size(), ")."));
  }
  if (buffers.empty()) {
    return absl::InvalidArgumentError(
        "Cannot make a DeviceArray with no data!");
  }
  auto* client = devices[0].device()->client();
  for (Device dev : devices) {
    DCHECK(dev.device()->client() == client)
        << "All devices must be from the same client!";
  }
  absl::InlinedVector<std::unique_ptr<xla::PjRtBuffer>, 1> shards;
  for (size_t i = 0; i < buffers.size(); ++i) {
    auto buf = internal::ToAbslStatusOr(client->BufferFromHostBuffer(
        buffers[i], internal::PrimitiveTypeToXla(type), dims, byte_strides,
        // Must be strict in order to be safe.
        xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
        devices[i].device()));
    if (!buf.ok()) return buf.status();
    shards.emplace_back(std::move(buf).value());
  }
  return DeviceArray(std::move(shards));
}

static Array LiteralToArray(std::shared_ptr<xla::Literal> literal) {
  xla::Literal& m = *literal;
  Shape shape(m.shape().dimensions().begin(), m.shape().dimensions().end());
  ShapedArray aval{internal::PrimitiveTypeFromXla(m.shape().element_type()),
                   std::move(shape)};
  return Array(aval, m.untyped_data(), {}, literal);
}

absl::StatusOr<absl::InlinedVector<Array, 1>> DeviceArray::ToArrays() const {
  absl::InlinedVector<Array, 1> out;
  for (auto& buf : buffers_) {
    auto literal = std::make_unique<xla::Literal>(
        xla::ShapeUtil::DeviceShapeToHostShape(buf->on_device_shape()));
    auto status = internal::ToAbslStatus(buf->ToLiteralSync(literal.get()));
    if (!status.ok()) return status;
    out.push_back(LiteralToArray(std::move(literal)));
  }
  return out;
}

void DeviceArray::ToArrayAsync(
    size_t idx, std::function<void(absl::StatusOr<Array>)> on_ready) const {
  if (idx >= size()) {
    on_ready(absl::InvalidArgumentError(absl::StrCat(
        "Index ", idx, " of DeviceArray out of range [0,", size(), ")")));
  }
  xla::PjRtBuffer* buf = buffers_[idx].get();
  auto literal = std::make_shared<xla::Literal>(
      xla::ShapeUtil::DeviceShapeToHostShape(buf->on_device_shape()));
  buf->ToLiteral(literal.get(), [=](xla::Status s) {
    if (s.ok()) {
      on_ready(LiteralToArray(literal));
    } else {
      on_ready(internal::ToAbslStatus(s));
    }
  });
}

absl::InlinedVector<xla::PjRtBuffer*, 1> DeviceArray::buffers() const {
  absl::InlinedVector<xla::PjRtBuffer*, 1> out;
  out.reserve(buffers_.size());
  for (auto& buf : buffers_) {
    out.push_back(buf.get());
  }
  return out;
}

std::vector<std::unique_ptr<xla::PjRtBuffer>> DeviceArray::ConsumeShards() {
  std::vector<std::unique_ptr<xla::PjRtBuffer>> out;
  for (auto& buf : buffers_) {
    out.push_back(std::move(buf));
  }
  buffers_.clear();
  return out;
}

Executable::Executable(std::unique_ptr<xla::PjRtExecutable> executable)
    : executable_(std::move(executable)) {}
Executable::~Executable() {}

absl::StatusOr<std::vector<DeviceArray>> Executable::Eval(
    absl::Span<const DeviceArray*> args) const {
  std::vector<DeviceArray> out;
  std::vector<xla::PjRtBuffer*> raw_args(args.size());
  auto devices = executable_->addressable_devices();
  for (size_t j = 0; j < args.size(); ++j) {
    if (args[j]->size() != devices.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Argument size mismatch: ", args[j]->size(), " vs ", devices.size()));
    }
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = true;
  for (size_t i = 0; i < devices.size(); ++i) {
    for (size_t j = 0; j < args.size(); ++j) {
      raw_args[j] = args[j]->buffers_[i].get();
    }
    auto result_statusor =
        executable_->ExecuteSharded(raw_args, devices[i], execute_options);
    if (!result_statusor.ok())
      return internal::ToAbslStatus(result_statusor.status());
    auto result = std::move(result_statusor).ValueOrDie();
    if (i == 0) {
      out.reserve(result.size());
      for (size_t j = 0; j < result.size(); ++j) {
        DeviceArray tmp;
        tmp.buffers_.reserve(devices.size());
        tmp.buffers_.push_back(std::move(result[j]));
        out.push_back(std::move(tmp));
      }
    } else {
      for (size_t j = 0; j < result.size(); ++j) {
        out[j].buffers_.push_back(std::move(result[j]));
      }
    }
  }
  return out;
}

}  // namespace aux
